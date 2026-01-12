import torch
from pathlib import Path
from typing import Optional, Dict, Any

from src.models.model_factory import ModelFactory
import wandb
from src.utils.config import load_config, get_model_config, get_training_config, get_data_config
from src.utils.utils import setup_logger, seed, get_device
from src.utils.wandb import login_wandb


def train_model(config_path: str = "configs/train_config.yaml") -> Dict[str, Any]:

    # 설정 로드
    config = load_config(config_path)
    model_cfg = get_model_config(config)
    train_cfg = get_training_config(config)
    data_cfg = get_data_config(config)
    exp_cfg = config.get('experiment', {})
    wandb_cfg = config.get('wandb', {})

    # 학습 파라미터 우선 설정 (로거 경로에 필요)
    project_dir = train_cfg.get('save_dir', 'runs/detect')
    exp_name = exp_cfg.get('name', 'exp')
    log_dir = Path(project_dir) / exp_name

    # 로거 설정
    logger = setup_logger(log_dir=log_dir)
    logger.info("Starting training...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Logs will be saved to: {log_dir}")

    # 시드 고정
    seed_val = exp_cfg.get('seed', 42)
    seed(seed_val)
    logger.info(f"Seed fixed to {seed_val}")

    # 디바이스 설정
    device = get_device(verbose=False)
    logger.info(f"Using device: {device}")

    # 모델 생성
    logger.info(f"Creating model: {model_cfg['type']}{model_cfg['size']}")
    model = ModelFactory.create_model(
        model_type=model_cfg['type'],
        size=model_cfg['size'],
        pretrained=model_cfg.get('pretrained', True)
    )

    # W&B 설정 및 초기화
    if wandb_cfg.get('enabled', False):
        login_wandb()
        run_name = model.model_name
        wandb.init(
            project=wandb_cfg.get('project', 'PCB_Detection'),
            name=run_name,
            tags=wandb_cfg.get('tags', []),
            config=config
        )
        logger.info(f"W&B initialized for project='{wandb_cfg.get('project')}', name='{run_name}'")

    # 학습 파라미터 설정
    train_params = {
        'data': data_cfg['data_yaml'],
        'epochs': train_cfg['epochs'],
        'batch': train_cfg['batch_size'],
        'imgsz': data_cfg['img_size'],
        'patience': train_cfg.get('patience', 10),
        'device': device,
        'workers': train_cfg.get('workers', 8),
        'save': True,
        'save_period': train_cfg.get('save_period', 10),
        'project': project_dir,
        'name': exp_name,
    }

    # 주석 처리된 augmentation 및 optimizer 설정은 더 이상 여기서 읽지 않습니다.
    # 해당 파라미터가 train_params에 전달되지 않으면 YOLOv8의 기본값이 사용됩니다.

    logger.info(f"Training parameters: {train_params}")

    # 학습 시작
    logger.info("Starting model training...")
    try:
        results = model.train(**train_params)
        logger.info("Training completed successfully")
        logger.info(f"Best model saved at: {train_params['project']}/{train_params['name']}/weights/best.pt")

        # 결과 반환
        return {
            'model': model,
            'results': results,
            'config': config
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # 학습 실행
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        result = train_model("configs/train_config.yaml")
        logger = logging.getLogger(__name__)
        logger.info("Training completed")
        logger.info(f"Results: {result['results']}")
    except Exception as e:
        logging.error(f"Training execution failed: {e}")
