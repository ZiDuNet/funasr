"""FastAPI 应用工厂 — 组装所有路由、模型加载、任务 Worker"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.models.registry import ModelRegistry
from server.models.config import DATA_DIR
from server.core.task_manager import TaskManager
from server.api import tasks as tasks_api

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 确保数据目录
    os.makedirs(DATA_DIR, exist_ok=True)

    # 加载模型
    registry = ModelRegistry.get_instance()
    app.state.registry = registry
    logger.info(f"设备: {registry.device}")
    registry.preload_all()

    # 启动异步任务 Worker
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
        description="统一语音识别：OpenAI API + HTTP REST + WebSocket 流式 + MCP + 声纹管理",
        version="1.0.0",
        lifespan=lifespan,
    )

    from server.api.openai_api import router as openai_router
    from server.api.http_rest import router as rest_router
    from server.api.tasks import router as task_router
    from server.api.speakers import router as speaker_router
    from server.api.websocket import register_ws_endpoint
    from server.mcp_server import get_mcp_app

    app.include_router(openai_router)
    app.include_router(rest_router)
    app.include_router(task_router)
    app.include_router(speaker_router)

    register_ws_endpoint(app)

    mcp_app = get_mcp_app()
    app.mount("/mcp", mcp_app)

    # Web UI
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.isdir(web_dir):
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
