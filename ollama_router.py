#!/usr/bin/env python3
# Copyright 2026 Satisfanly Ltd
#
# OKO OS is a product of Satisfanly Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transparent Ollama router.

- Discovers models from configured Ollama-compatible upstreams via /api/tags.
- Refreshes discovery periodically (30 seconds by default).
- Routes requests by model name.
- Forwards the original HTTP method, path, query string, headers and raw body.
- Streams upstream responses without buffering/reformatting them.
- Merges /api/tags and /api/ps across upstreams.
- Merges /v1/models across upstreams.
- Uses a very large upstream response timeout (12 hours by default).

This is intentionally NOT an Ollama/OpenAI translator.

Dependency:
    aiohttp

Example:
    python3 ollama_transparent_router.py \
        --listen 0.0.0.0:11436 \
        --endpoints 127.0.0.1:11434 127.0.0.1:11435

Then:
    OLLAMA_HOST=127.0.0.1:11436 ollama ls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from multidict import CIMultiDict


LOG = logging.getLogger("ollama-router")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Paths where the request body normally contains a model identifier.
MODEL_FIELDS_BY_PATH: dict[str, tuple[str, ...]] = {
    "/api/chat": ("model",),
    "/api/generate": ("model",),
    "/api/embed": ("model",),
    "/api/embeddings": ("model",),
    "/api/show": ("model", "name"),
    "/api/delete": ("model", "name"),
    "/api/pull": ("model", "name"),
    "/api/push": ("model", "name"),
    "/api/create": ("model", "name"),
    "/api/copy": ("source",),
    "/v1/chat/completions": ("model",),
    "/v1/completions": ("model",),
    "/v1/embeddings": ("model",),
}

# A successful request to one of these can change model inventory.
MODEL_MUTATION_PATHS = {
    "/api/create",
    "/api/copy",
    "/api/delete",
    "/api/pull",
}


def normalize_endpoint(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty endpoint")

    if "://" not in value:
        value = "http://" + value

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported endpoint scheme in {value!r}")
    if not parsed.netloc:
        raise ValueError(f"invalid endpoint {value!r}")

    return value.rstrip("/")


def parse_listen(value: str) -> tuple[str, int]:
    value = value.strip()
    if not value:
        raise ValueError("empty listen address")

    # Support [::1]:11434.
    if value.startswith("["):
        end = value.find("]")
        if end < 0 or end + 2 > len(value) or value[end + 1] != ":":
            raise ValueError(f"invalid listen address: {value!r}")
        host = value[1:end]
        port_text = value[end + 2 :]
    else:
        if ":" not in value:
            raise ValueError("listen address must be HOST:PORT")
        host, port_text = value.rsplit(":", 1)

    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError("listen port must be between 1 and 65535")
    return host, port


def filter_request_headers(headers: "web.BaseRequest.headers") -> CIMultiDict[str]:
    out = CIMultiDict()
    for name, value in headers.items():
        lower = name.lower()
        if lower in HOP_BY_HOP_HEADERS or lower in {"host", "content-length"}:
            continue
        out.add(name, value)
    return out


def filter_response_headers(headers: "aiohttp.typedefs.LooseHeaders") -> CIMultiDict[str]:
    out = CIMultiDict()
    for name, value in headers.items():
        lower = name.lower()
        if lower in HOP_BY_HOP_HEADERS:
            continue
        out.add(name, value)
    return out


def extract_model(path: str, body: bytes) -> str | None:
    fields = MODEL_FIELDS_BY_PATH.get(path)
    if not fields or not body:
        return None

    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(obj, dict):
        return None

    for field in fields:
        value = obj.get(field)
        if isinstance(value, str) and value:
            return value

    return None


@dataclass
class EndpointState:
    url: str
    models: dict[str, dict[str, Any]]
    last_discovery_ok: bool = False


class OllamaRouter:
    def __init__(
        self,
        endpoints: list[str],
        *,
        timeout_seconds: float,
        discovery_interval: float,
        discovery_timeout: float,
    ) -> None:
        self.endpoints = [EndpointState(url=e, models={}) for e in endpoints]
        self.timeout_seconds = timeout_seconds
        self.discovery_interval = discovery_interval
        self.discovery_timeout = discovery_timeout

        self.session: aiohttp.ClientSession | None = None
        self.discovery_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()

        self.model_to_endpoint: dict[str, str] = {}
        self.merged_models: list[dict[str, Any]] = []
        self.discovery_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=min(60.0, self.timeout_seconds),
            sock_connect=min(60.0, self.timeout_seconds),
            sock_read=self.timeout_seconds,
        )
        connector = aiohttp.TCPConnector(
            limit=0,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            auto_decompress=False,
        )

        await self.discover()
        self.discovery_task = asyncio.create_task(
            self._discovery_loop(), name="ollama-model-discovery"
        )

    async def stop(self) -> None:
        if self.discovery_task is not None:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
            self.discovery_task = None

        if self.session is not None:
            await self.session.close()
            self.session = None

    async def _discovery_loop(self) -> None:
        while True:
            await asyncio.sleep(self.discovery_interval)
            try:
                await self.discover()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("periodic model discovery failed")

    async def _discover_endpoint(
        self, endpoint: EndpointState
    ) -> tuple[EndpointState, dict[str, dict[str, Any]] | None]:
        assert self.session is not None

        url = endpoint.url + "/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(
                total=self.discovery_timeout,
                connect=min(30.0, self.discovery_timeout),
                sock_connect=min(30.0, self.discovery_timeout),
                sock_read=self.discovery_timeout,
            )
            async with self.session.get(url, timeout=timeout) as resp:
                raw = await resp.read()
                if resp.status != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status}: {raw[:500].decode(errors='replace')}"
                    )

            payload = json.loads(raw)
            models = payload.get("models")
            if not isinstance(models, list):
                raise RuntimeError("response does not contain a models list")

            result: dict[str, dict[str, Any]] = {}
            for item in models:
                if not isinstance(item, dict):
                    continue

                # Current Ollama normally has both "name" and "model".
                names: list[str] = []
                for key in ("name", "model"):
                    value = item.get(key)
                    if isinstance(value, str) and value and value not in names:
                        names.append(value)

                for name in names:
                    result[name] = item

            LOG.info(
                "discovered %d model name(s) from %s",
                len(result),
                endpoint.url,
            )
            return endpoint, result

        except Exception as exc:
            LOG.warning("discovery failed for %s: %s", endpoint.url, exc)
            return endpoint, None

    async def discover(self) -> None:
        """
        Refresh model inventory.

        If an endpoint temporarily fails discovery, retain its previous model
        list instead of deleting it. This prevents a slow/flaky /api/tags call
        from immediately destroying routing state.
        """
        async with self.discovery_lock:
            results = await asyncio.gather(
                *(self._discover_endpoint(ep) for ep in self.endpoints)
            )

            async with self.state_lock:
                for endpoint, discovered in results:
                    if discovered is None:
                        endpoint.last_discovery_ok = False
                        continue
                    endpoint.models = discovered
                    endpoint.last_discovery_ok = True

                # First configured endpoint wins if a model exists in multiple
                # upstreams. This makes routing deterministic.
                model_map: dict[str, str] = {}
                merged_by_name: dict[str, dict[str, Any]] = {}

                for endpoint in self.endpoints:
                    for name, item in endpoint.models.items():
                        model_map.setdefault(name, endpoint.url)

                        canonical = item.get("name") or item.get("model") or name
                        if isinstance(canonical, str):
                            merged_by_name.setdefault(canonical, item)

                self.model_to_endpoint = model_map
                self.merged_models = list(merged_by_name.values())

                if model_map:
                    LOG.info(
                        "model map refreshed: %d model name(s), %d visible model(s)",
                        len(model_map),
                        len(self.merged_models),
                    )
                else:
                    LOG.warning("model map is empty")

    async def resolve_model(self, model: str) -> str | None:
        async with self.state_lock:
            endpoint = self._resolve_model_locked(model)
        if endpoint is not None:
            return endpoint

        # A model may have appeared since the last periodic refresh.
        await self.discover()

        async with self.state_lock:
            return self._resolve_model_locked(model)

    def _resolve_model_locked(self, model: str) -> str | None:
        endpoint = self.model_to_endpoint.get(model)
        if endpoint is not None:
            return endpoint

        # Native Ollama treats a missing tag as :latest.
        if ":" not in model:
            endpoint = self.model_to_endpoint.get(model + ":latest")
            if endpoint is not None:
                return endpoint

        # Also permit a discovered "foo" to satisfy "foo:latest".
        if model.endswith(":latest"):
            endpoint = self.model_to_endpoint.get(model[:-7])
            if endpoint is not None:
                return endpoint

        return None

    async def default_endpoint(self) -> str:
        # Prefer an endpoint that succeeded most recently, otherwise preserve
        # configured ordering.
        async with self.state_lock:
            for endpoint in self.endpoints:
                if endpoint.last_discovery_ok:
                    return endpoint.url
            return self.endpoints[0].url

    async def tags(self, request: web.Request) -> web.Response:
        async with self.state_lock:
            payload = {"models": list(self.merged_models)}
        return web.json_response(payload)

    async def _fetch_json_from_all(
        self, path: str
    ) -> list[tuple[str, int, bytes, dict[str, Any] | None]]:
        assert self.session is not None

        async def one(endpoint: EndpointState):
            url = endpoint.url + path
            try:
                async with self.session.get(url) as resp:
                    raw = await resp.read()
                    parsed: dict[str, Any] | None = None
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            parsed = obj
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    return endpoint.url, resp.status, raw, parsed
            except Exception as exc:
                LOG.warning("%s failed for %s: %s", path, endpoint.url, exc)
                return endpoint.url, 0, str(exc).encode(), None

        return await asyncio.gather(*(one(ep) for ep in self.endpoints))

    async def ps(self, request: web.Request) -> web.Response:
        results = await self._fetch_json_from_all("/api/ps")

        merged: list[Any] = []
        seen: set[str] = set()

        for _endpoint, status, _raw, obj in results:
            if status != 200 or not obj:
                continue
            models = obj.get("models")
            if not isinstance(models, list):
                continue

            for item in models:
                if not isinstance(item, dict):
                    merged.append(item)
                    continue

                key = str(
                    item.get("name")
                    or item.get("model")
                    or item.get("digest")
                    or json.dumps(item, sort_keys=True, default=str)
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)

        return web.json_response({"models": merged})

    async def v1_models(self, request: web.Request) -> web.Response:
        results = await self._fetch_json_from_all("/v1/models")

        merged: list[Any] = []
        seen: set[str] = set()

        for _endpoint, status, _raw, obj in results:
            if status != 200 or not obj:
                continue
            data = obj.get("data")
            if not isinstance(data, list):
                continue

            for item in data:
                if not isinstance(item, dict):
                    merged.append(item)
                    continue
                key = str(item.get("id") or json.dumps(item, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)

        return web.json_response({"object": "list", "data": merged})

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None

        # Read bytes once for routing, then forward the exact same bytes.
        body = await request.read()
        path = request.path
        model = extract_model(path, body)

        if model:
            endpoint = await self.resolve_model(model)
            if endpoint is None:
                # Unknown/new model (e.g. pull/create). Let the first healthy
                # upstream return its native Ollama response/error.
                endpoint = await self.default_endpoint()
                LOG.info(
                    "model %r not discovered; forwarding %s to default %s",
                    model,
                    path,
                    endpoint,
                )
            else:
                LOG.debug("routing model %r to %s", model, endpoint)
        else:
            endpoint = await self.default_endpoint()

        upstream_url = endpoint + request.rel_url.path_qs
        headers = filter_request_headers(request.headers)

        try:
            async with self.session.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                data=body if body else None,
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=filter_response_headers(upstream.headers),
                )
                await response.prepare(request)

                try:
                    async for chunk in upstream.content.iter_any():
                        if chunk:
                            await response.write(chunk)
                    await response.write_eof()
                except (
                    ConnectionResetError,
                    BrokenPipeError,
                    asyncio.CancelledError,
                ):
                    upstream.close()
                    raise

                # Refresh immediately after a successful model mutation.
                if path in MODEL_MUTATION_PATHS and upstream.status < 400:
                    asyncio.create_task(self.discover())

                return response

        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            LOG.error("upstream request failed: %s -> %s: %s", path, endpoint, exc)
            raise web.HTTPBadGateway(
                text=f"upstream Ollama request failed: {exc}\n"
            ) from exc

    async def handle(self, request: web.Request) -> web.StreamResponse:
        # The two inventory endpoints are the only native Ollama endpoints
        # that need aggregation rather than single-backend forwarding.
        if request.path == "/api/tags" and request.method in {"GET", "HEAD"}:
            if request.method == "HEAD":
                return web.Response(status=200)
            return await self.tags(request)

        if request.path == "/api/ps" and request.method in {"GET", "HEAD"}:
            if request.method == "HEAD":
                return web.Response(status=200)
            return await self.ps(request)

        if request.path == "/v1/models" and request.method in {"GET", "HEAD"}:
            if request.method == "HEAD":
                return web.Response(status=200)
            return await self.v1_models(request)

        # Everything else, including GET/HEAD /, /api/version, /api/show,
        # /api/chat, /api/generate, /api/embed, management operations, and
        # future Ollama endpoints is forwarded rather than translated.
        return await self.proxy(request)


def build_app(router: OllamaRouter) -> web.Application:
    app = web.Application(client_max_size=1024**4)  # 1 TiB; effectively no practical body limit.

    async def on_startup(_app: web.Application) -> None:
        await router.start()

    async def on_cleanup(_app: web.Application) -> None:
        await router.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Catch-all preserves all present/future Ollama endpoint paths.
    app.router.add_route("*", "/{tail:.*}", router.handle)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transparent model-aware router for multiple Ollama servers"
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:11436",
        metavar="HOST:PORT",
        help="listen address (default: 127.0.0.1:11436)",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        required=True,
        metavar="ADDR",
        help=(
            "Ollama upstreams, e.g. "
            "--endpoints 127.0.0.1:11434 127.0.0.1:11435"
        ),
    )
    parser.add_argument(
        "--rediscover-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="model rediscovery interval (default: 30)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="timeout for each /api/tags discovery request (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12 * 60 * 60,
        metavar="SECONDS",
        help="upstream response/read timeout (default: 43200 = 12 hours)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rediscover_interval <= 0:
        raise SystemExit("--rediscover-interval must be > 0")
    if args.discovery_timeout <= 0:
        raise SystemExit("--discovery-timeout must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")

    try:
        endpoints = [normalize_endpoint(v) for v in args.endpoints]
        host, port = parse_listen(args.listen)
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc

    # Remove exact duplicate endpoint arguments while preserving order.
    endpoints = list(dict.fromkeys(endpoints))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    LOG.info("listening on %s:%d", host, port)
    LOG.info("upstreams: %s", ", ".join(endpoints))
    LOG.info(
        "rediscovery every %.1fs; upstream timeout %.0fs",
        args.rediscover_interval,
        args.timeout,
    )

    router = OllamaRouter(
        endpoints,
        timeout_seconds=args.timeout,
        discovery_interval=args.rediscover_interval,
        discovery_timeout=args.discovery_timeout,
    )
    app = build_app(router)

    web.run_app(
        app,
        host=host,
        port=port,
        access_log=LOG,
        print=None,
    )


if __name__ == "__main__":
    main()

