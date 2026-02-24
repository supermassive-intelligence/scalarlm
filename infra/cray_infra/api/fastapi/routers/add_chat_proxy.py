from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import RedirectResponse

CHAT_UI_BASE = "http://127.0.0.1:3000/chat"
CHAT_UI_ORIGIN = "http://localhost:3000"



def add_chat_proxy(app: FastAPI):
    @app.api_route("/chat/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def proxy_chat(request: Request, path: str):
        url = f"{CHAT_UI_BASE}/{path}"

        if request.query_params:
            url += f"?{request.query_params}"

        headers = {}
        for k, v in request.headers.items():
            key = k.lower()
            if key == "host":
                # must match CHAT_UI_ORIGIN's host
                headers[k] = CHAT_UI_ORIGIN.split("://")[-1]
            elif key == "accept-encoding":
                continue
            elif key == "origin":
                headers[k] = CHAT_UI_ORIGIN
            elif key == "referer":
                # Rewrite e.g. http://localhost:8000/chat/... -> http://localhost:3000/chat/...
                headers[k] = v.replace(str(request.base_url).rstrip("/"), CHAT_UI_ORIGIN)
            else:
                headers[k] = v
        body = await request.body()

        session = get_global_session()

        async def stream_response(resp):
            async for chunk in resp.content.iter_any():
                yield chunk

        resp = await session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body,
            ssl=False,
        )

        excluded_response_headers = {"content-length", "transfer-encoding", "content-encoding"}

        return StreamingResponse(
            stream_response(resp),
            status_code=resp.status,
            headers={
                k: v for k, v in resp.headers.items()
                if k.lower() not in excluded_response_headers
            },
        )

    @app.get("/chat")
    async def chat_root():
        return RedirectResponse(url="/chat/")
