"""FastAPI 应用工厂"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from server.models.registry import ModelRegistry
from server.models.config import DATA_DIR, PRELOAD_ALL, ENABLE_STREAMING, ENABLE_MCP, MODEL_NAME
from server.core.task_manager import TaskManager
from server.api import tasks as tasks_api

logger = logging.getLogger(__name__)

# ── API Token 认证中间件 ───────────────────────────
# .env 中设置 API_TOKEN=xxx 启用认证，留空或不设置则不认证
_API_TOKEN = os.environ.get("API_TOKEN", "").strip()


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Token 认证中间件

    - API_TOKEN 为空时不做认证（开发模式）
    - /health、/docs、静态文件不需要认证
    - API 路由通过 Authorization: Bearer <token> 认证
    - WebSocket 通过 ?token=xxx 查询参数认证
    """

    # 无需认证的路径
    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next):
        # 未配置 token = 不认证
        if not _API_TOKEN:
            return await call_next(request)

        path = request.url.path

        # 公开路径
        if path in self.PUBLIC_PATHS:
            return await call_next(request)

        # 静态文件（CSS/JS/HTML/图片）
        if (path.startswith(("/css/", "/js/")) or path == "/"
                or path.endswith((".html", ".css", ".js", ".png", ".ico", ".svg"))):
            return await call_next(request)

        # WebSocket：从查询参数获取 token
        if path == "/ws":
            token = request.query_params.get("token", "")
            if token == _API_TOKEN:
                return await call_next(request)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        # API 路由：从 Header 或查询参数获取 token
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.query_params.get("token", "")

        if token == _API_TOKEN:
            return await call_next(request)

        return JSONResponse(
            {"error": "Unauthorized", "detail": "Invalid or missing API token"},
            status_code=401,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)

    registry = ModelRegistry.get_instance()
    app.state.registry = registry
    logger.info(f"设备: {registry.device}, 模型: {MODEL_NAME}")

    if _API_TOKEN:
        logger.info(f"API Token 认证: 已启用")
    else:
        logger.warning("API Token 认证: 未启用（API_TOKEN 为空），建议生产环境设置")

    if PRELOAD_ALL:
        registry.preload()
    else:
        logger.info("懒加载模式，模型将在首次请求时加载")

    task_manager = TaskManager()
    await task_manager.start()
    tasks_api.set_task_manager(task_manager)
    app.state.task_manager = task_manager

    logger.info("FunASR All-in-One 服务就绪！")
    yield
    await task_manager.stop()
    registry.shutdown()
    logger.info("服务已关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FunASR All-in-One",
        description=f"统一语音识别 ({MODEL_NAME}) — OpenAI API + HTTP REST + WebSocket + MCP + 声纹",
        version="1.0.0", lifespan=lifespan,
    )

    # 允许所有来源（局域网/公网访问）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Token 认证（API_TOKEN 非空时生效）
    app.add_middleware(TokenAuthMiddleware)

    from server.api.openai_api import router as openai_router
    from server.api.http_rest import router as rest_router
    from server.api.tasks import router as task_router
    from server.api.speakers import router as speaker_router

    app.include_router(openai_router)
    app.include_router(rest_router)
    app.include_router(task_router)
    app.include_router(speaker_router)

    if ENABLE_STREAMING:
        from server.api.websocket import register_ws_endpoint
        register_ws_endpoint(app)

    if ENABLE_MCP:
        from server.mcp_server import get_mcp_app
        app.mount("/mcp", get_mcp_app())

    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.isdir(web_dir):
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
