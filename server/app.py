"""FastAPI 应用工厂"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.models.registry import ModelRegistry
from server.models.config import DATA_DIR, PRELOAD_ALL, ENABLE_STREAMING, ENABLE_MCP, MODEL_NAME
from server.core.task_manager import TaskManager
from server.api import tasks as tasks_api

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)

    registry = ModelRegistry.get_instance()
    app.state.registry = registry
    logger.info(f"设备: {registry.device}, 模型: {MODEL_NAME}")

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
