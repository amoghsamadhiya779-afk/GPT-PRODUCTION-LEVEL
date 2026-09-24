# app/security.py
"""Request-level security primitives for the serving API.

- client_ip:      rate-limit identity that is correct behind a reverse proxy
                  without letting clients spoof it.
- require_admin:  bearer-token guard for operator-only endpoints.
- SecurityMiddleware: request body size cap + defensive response headers.

All settings are read from the environment at call time, so tests (and
operators restarting with new env) never depend on import order.
"""

import hmac
import os

from fastapi import HTTPException, Request, status
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    """Identify the caller for rate limiting.

    By default this is the TCP peer address. Behind reverse proxies (Hugging
    Face Spaces, a load balancer, a CDN) the peer is the proxy itself, so every
    user would share one rate-limit bucket. Set TRUSTED_PROXY_HOPS to the
    number of proxies in front of the app: each appends the address it saw to
    X-Forwarded-For, so the entry that many places from the right was written
    by our outermost trusted proxy. Entries further left are client-supplied
    and must never be trusted -- taking the leftmost one (a common mistake)
    lets anyone bypass rate limits by sending a fake header.
    """
    hops = _int_env("TRUSTED_PROXY_HOPS", 0)
    if hops > 0:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return request.client.host if request.client else "unknown"


def require_admin(request: Request) -> None:
    """FastAPI dependency: allow only callers presenting the ADMIN_API_KEY.

    Admin endpoints are disabled outright (403) when no key is configured, so
    a deployment that never sets one can't be left accidentally open.
    """
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is disabled: ADMIN_API_KEY is not configured on the server.",
        )
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    # compare_digest: constant-time, so response timing doesn't leak the key.
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
]


class _BodyTooLarge(Exception):
    pass


class SecurityMiddleware:
    """Pure-ASGI middleware (safe for streaming responses, unlike
    BaseHTTPMiddleware) that caps request body size and adds security headers.

    Without a cap, uvicorn happily buffers an arbitrarily large JSON body into
    memory before validation ever runs -- one request can exhaust the RAM the
    model needs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = _int_env("MAX_REQUEST_BYTES", 2 * 1024 * 1024)
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None and (not content_length.isdigit() or int(content_length) > max_bytes):
            await self._reject(send, max_bytes)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:  # chunked uploads carry no Content-Length
                    raise _BodyTooLarge()
            return message

        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message["headers"] = list(message.get("headers", [])) + _SECURITY_HEADERS
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_headers)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send, max_bytes)

    @staticmethod
    async def _reject(send: Send, max_bytes: int) -> None:
        body = b'{"detail":"Request body too large (limit %d bytes)."}' % max_bytes
        await send({
            "type": "http.response.start",
            "status": status.HTTP_413_CONTENT_TOO_LARGE,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
            + _SECURITY_HEADERS,
        })
        await send({"type": "http.response.body", "body": body})
