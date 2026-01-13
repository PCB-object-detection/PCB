import os
import logging
import wandb
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)


def login_wandb() -> None:
    try:
        # 프로젝트 루트에서 .env 파일 찾기
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/utils/wandb.py -> PCB/
        env_file = project_root / ".env"

        # .env 파일 로드
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            logger.info(f"Loaded .env from: {env_file}")
        else:
            load_dotenv()  # 현재 디렉토리에서 찾기
            logger.warning(f".env file not found at: {env_file}")

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
            logger.info("W&B login successful.")
        else:
            logger.warning("WANDB_API_KEY not found in environment. Trying to use existing logged-in session.")

    except Exception as e:
        logger.error(f"Failed to login to W&B: {e}")
        pass