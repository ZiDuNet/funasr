"""FastAPI 应用工厂"""

import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
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
    - WebSocket 认证在 WebSocket endpoint 内通过 ?token=xxx 处理
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
        if path.startswith(("/css/", "/js/")) or path == "/":
            return await call_next(request)

        # API 路由：从 Header 或查询参数获取 token
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.query_params.get("token", "")

        if token == _API_TOKEN:
            return await call_next(request)

        return JSONResponse(
            {"error": "unauthorized", "detail": "API Token 缺失或错误"},
            status_code=401,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        mcp_app = getattr(app.state, "mcp_app", None)
        if mcp_app is not None and hasattr(mcp_app, "lifespan"):
            await stack.enter_async_context(mcp_app.lifespan(app))

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
        title="FunASR All-in-One 统一语音识别服务",
        description=f"统一语音识别 ({MODEL_NAME})：标准 API、OpenAI 兼容 API、实时流式、声纹组、MCP、原生 WebUI",
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
    from server.api.tasks import router as task_router
    from server.api.speakers import router as speaker_router

    app.include_router(openai_router)
    app.include_router(task_router)
    app.include_router(speaker_router)

    if ENABLE_STREAMING:
        from server.api.websocket import register_ws_endpoint
        register_ws_endpoint(app)

    if ENABLE_MCP:
        from server.mcp_server import get_mcp_app
        mcp_app = get_mcp_app()
        app.state.mcp_app = mcp_app
        app.mount("/mcp", mcp_app)

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        css_dir = web_dir / "css"
        js_dir = web_dir / "js"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="web-css")
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="web-js")

        @app.get("/", include_in_schema=False)
        async def root_webui():
            return FileResponse(web_dir / "index.html")

        logger.info("原生 WebUI 已挂载: /")
    else:
        logger.warning(f"原生 WebUI 目录不存在: {web_dir}")

    return app
