import os
import logging
import wandb
from dotenv import load_dotenv
from pathlib import Path
from src.utils.config import get_model_config, get_training_config, get_data_config

logger = logging.getLogger(__name__)

def login_wandb() -> None:
    """.env에서 API key 로드 후 W&B 로그인"""
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            logger.info(f"Loaded .env from: {env_file}")
        else:
            load_dotenv()
            logger.warning(f".env file not found at: {env_file}")

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
            logger.info("W&B login successful.")
        else:
            logger.warning("WANDB_API_KEY not found. Trying existing session.")
    except Exception as e:
        logger.error(f"Failed to login to W&B: {e}")


def init_wandb(config: dict, model, device: str, gpu_name: str, exp_name: str):
    """config에서 W&B 정보 읽고 wandb.init 실행 + YOLO 연동"""
    wandb_cfg = config.get('wandb', {})
    if not wandb_cfg.get('enabled', False):
        logger.info("W&B disabled in config.")
        return

    login_wandb()
    model_cfg = get_model_config(config)
    train_cfg = get_training_config(config)
    data_cfg = get_data_config(config)
    run_name = f"{model_cfg['type']}{model_cfg['size']}-{wandb_cfg.get('suffix', 'exp')}"

    # YOLO wandb 연동
    if model_cfg['type'].lower().startswith('yolo'):
        try:
            from ultralytics import settings
            settings.update({'wandb': True})
            logger.info("Ultralytics wandb integration enabled")
        except Exception as e:
            logger.warning(f"Failed to update ultralytics settings: {e}")

    # W&B config dict
    wandb_config = {
        'model_type': model_cfg['type'],
        'model_size': model_cfg['size'],
        'pretrained': model_cfg.get('pretrained', True),
        'epochs': train_cfg['epochs'],
        'batch_size': train_cfg['batch_size'],
        'img_size': data_cfg['img_size'],
        'patience': train_cfg.get('patience', 10),
        'device': device,
        'gpu_name': gpu_name,
        'freeze': train_cfg.get('freeze', 0),
        'optimizer': train_cfg.get('optimizer', 'auto'),
        'lr0': train_cfg.get('lr0', 0.01),
        'lrf': train_cfg.get('lrf', 0.01),
        'momentum': train_cfg.get('momentum', 0.937),
        'weight_decay': train_cfg.get('weight_decay', 0.0005),
        'dropout': train_cfg.get('dropout', 0.0),
        'mosaic': train_cfg.get('mosaic', 0.0),
        'mixup': train_cfg.get('mixup', 0.0),
        'multi_scale': train_cfg.get('multi_scale', False),
        'box': train_cfg.get('box', 7.5),
        'cls': train_cfg.get('cls', 0.5),
        'run_name': exp_name,
    }

    # W&B init
    wandb.init(
        project=wandb_cfg.get('project', 'PCB_Detection'),
        name=run_name,
        tags=wandb_cfg.get('tags', []),
        config=wandb_config
    )
    logger.info(f"W&B initialized: project={wandb_cfg.get('project')}, run_name={run_name}")


def wandb_log_callback(trainer):
    """YOLO Trainer용 epoch 끝날 때마다 loss, metrics 실시간 로깅"""
    if wandb.run is None:
        return

    metrics = {}
    # Train loss
    if hasattr(trainer, 'tloss'):
        metrics['train_loss'] = float(trainer.tloss)
    # Validation metrics
    if hasattr(trainer, 'metrics'):
        metrics.update({f"val_{k}": float(v) for k, v in trainer.metrics.items()})

    wandb.log(metrics, step=getattr(trainer, 'epoch', 0))
    logger.info(f"W&B logged metrics at epoch {getattr(trainer, 'epoch', 0)}: {metrics}")


def log_best_metrics(model, results):
    """학습 종료 후 W&B summary에 기록"""
    if wandb.run is None:
        return
    try:
        trainer = getattr(model, 'trainer', None)
        best_metrics = {}

        if hasattr(results, 'box'):
            best_metrics = {
                'best/mAP50': float(results.box.map50),
                'best/mAP50-95': float(results.box.map),
                'best/precision': float(results.box.mp),
                'best/recall': float(results.box.mr),
                'best/fitness': float(getattr(results, 'fitness', 0.0)),
            }

        for key, value in best_metrics.items():
            wandb.run.summary[key] = value

        if trainer:
            stopped_epoch = trainer.epoch + 1
            best_epoch = getattr(trainer, 'best_epoch', stopped_epoch)
        else:
            stopped_epoch = 0
            best_epoch = 0

        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['stopped_epoch'] = stopped_epoch
        wandb.config.update({'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch}, allow_val_change=True)

        logger.info(f"Logged best metrics to W&B: {best_metrics}, best_epoch={best_epoch}, stopped_epoch={stopped_epoch}")
    except Exception as e:
        logger.warning(f"Failed to log best metrics: {e}")