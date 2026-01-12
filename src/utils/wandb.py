import os
import logging
import wandb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def login_wandb() -> None:
    """
    .env 파일에서 WANDB_API_KEY를 로드하여 Weights & Biases에 로그인합니다.
    """
    try:
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            logger.info("W&B login successful.")
        else:
            logger.warning("WANDB_API_KEY not found. Trying to use existing logged-in session.")

    except Exception as e:
        logger.error(f"Failed to login to W&B: {e}")
        pass