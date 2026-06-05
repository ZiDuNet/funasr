"""FunASR All-in-One 入口"""

import logging
import uvicorn

from server.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import os
    host = os.environ.get("FUNASR_HOST", "0.0.0.0")
    port = int(os.environ.get("FUNASR_PORT", "17767"))

    logger.info(f"启动 FunASR All-in-One @ http://{host}:{port}")
    logger.info(f"  OpenAI API:  http://{host}:{port}/v1/audio/transcriptions")
    logger.info(f"  HTTP REST:   http://{host}:{port}/recognition")
    logger.info(f"  WebSocket:   ws://{host}:{port}/ws")
    logger.info(f"  MCP:         http://{host}:{port}/mcp")
    logger.info(f"  Web UI:      http://{host}:{port}/")
    logger.info(f"  API Docs:    http://{host}:{port}/docs")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
