import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any
from src.models.model_factory import ModelFactory
from src.utils.config import load_config, get_model_config, get_training_config, get_data_config
from src.utils.utils import setup_logger, seed, get_device
from src.utils.wandb_setup import login_wandb, init_wandb, wandb_log_callback, log_best_metrics


def train_model(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    # 설정 로드
    config = load_config(config_path)
    model_cfg = get_model_config(config)
    train_cfg = get_training_config(config)
    data_cfg = get_data_config(config)
    exp_cfg = config.get('experiment', {})
    wandb_cfg = config.get('wandb', {})

    # 실험명 생성
    model_name = f"{model_cfg['type']}{model_cfg['size']}"
    exp_suffix = exp_cfg.get('suffix', 'exp')
    exp_name = f"{model_name}-{exp_suffix}"

    # 프로젝트 루트
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent if config_file.parent.name == "configs" else Path.cwd()
    os.chdir(project_root)

    # 로그 설정
    log_dir = project_root / "logs" / exp_name
    logger = setup_logger(log_dir=log_dir)
    logger.info("Starting training...")

    # 시드
    seed(seed_val := exp_cfg.get('seed', 42))
    logger.info(f"Seed fixed to {seed_val}")

    # 디바이스
    device = get_device(verbose=False)
    logger.info(f"Using device: {device}")

    # GPU 이름
    gpu_name = "CPU"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
        except Exception as e:
            logger.warning(f"Failed to get GPU name: {e}")

    # 모델 생성
    checkpoint = model_cfg.get('checkpoint', None)
    if checkpoint:
        logger.info(f"Loading model from checkpoint: {checkpoint}")
        model = ModelFactory.create_model(
            model_type=model_cfg['type'],
            size=model_cfg['size'],
            pretrained=False,
            weights_path=checkpoint
        )
    else:
        model = ModelFactory.create_model(
            model_type=model_cfg['type'],
            size=model_cfg['size'],
            pretrained=model_cfg.get('pretrained', True)
        )

    # Fine-tuning: backbone freeze
    if checkpoint:
        freeze_layers = train_cfg.get('freeze', 10)
        logger.info(f"Freezing first {freeze_layers} layers for fine-tuning")
        for i, (_, param) in enumerate(model.model.named_parameters()):
            if i < freeze_layers * 2:
                param.requires_grad = False

    # data_yaml 절대경로
    data_yaml_path = Path(data_cfg['data_yaml'])
    if not data_yaml_path.is_absolute():
        data_yaml_path = (project_root / data_yaml_path).resolve()

    output_dir = Path(train_cfg.get('save_dir', 'runs/detect'))
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    # W&B 초기화
    if wandb_cfg.get('enabled', False):
        login_wandb()
        init_wandb(config, model, device=str(device), gpu_name=gpu_name, exp_name=exp_name)

        # YOLO Trainer callback 등록
        if hasattr(model, 'add_callback'):
            model.add_callback('on_fit_epoch_end', wandb_log_callback)

    # 학습 파라미터
    train_params = {
        'data': str(data_yaml_path),
        'epochs': train_cfg['epochs'],
        'batch': train_cfg['batch_size'],
        'imgsz': data_cfg['img_size'],
        'patience': train_cfg.get('patience', 10),
        'device': device,
        'workers': train_cfg.get('workers', 8),
        'project': str(output_dir),
        'name': exp_name,
        'val': True,
        'optimizer': train_cfg.get('optimizer', 'auto'),
        'lr0': train_cfg.get('lr0', 0.01),
        'lrf': train_cfg.get('lrf', 0.01),
        'momentum': train_cfg.get('momentum', 0.937),
        'weight_decay': train_cfg.get('weight_decay', 0.0005),
        'mosaic': train_cfg.get('mosaic', 1.0),
        'mixup': train_cfg.get('mixup', 0.0),
        'multi_scale': train_cfg.get('multi_scale', False),
    }

    logger.info(f"Training parameters: {train_params}")

    # 학습
    try:
        results = model.train(**train_params)
        logger.info("Training completed successfully")
        logger.info(f"Best model saved at: {output_dir}/{exp_name}/weights/best.pt")

        # W&B best metric 기록
        if wandb_cfg.get('enabled', False):
            log_best_metrics(model, results)

        return {'model': model, 'results': results, 'config': config}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        result = train_model("configs/config.yaml")
        logger = logging.getLogger(__name__)
        logger.info("Training completed")
        logger.info(f"Results: {result['results']}")
    except Exception as e:
        logging.error(f"Training execution failed: {e}")