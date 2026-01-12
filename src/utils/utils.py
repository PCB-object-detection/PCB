import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional


def seed(seed: int):
    """random seed"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(verbose: bool = True) -> torch.device:
    """CUDA, MPS, CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using device: {device}")

    return device


def setup_logger(
    name: str = "PCB_Detection",
    log_dir: Optional[str] = None,
    log_level: str = "INFO"
) -> logging.Logger:

    # 로그 디렉토리 설정
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 핸들러 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger