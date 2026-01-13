import os
import logging
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

    # 실험명 자동 생성: {모델명}-{suffix}
    model_name = f"{model_cfg['type']}{model_cfg['size']}"  # 예: 'yolo11n'
    exp_suffix = exp_cfg.get('suffix', 'exp')  # config의 suffix 또는 'exp'
    exp_name = f"{model_name}-{exp_suffix}"  # 예: 'yolo11n-base'

    # 학습 파라미터 우선 설정
    project_dir = train_cfg.get('save_dir', 'runs/detect')

    # 프로젝트 루트 찾기 (config 파일 위치 기준)
    config_file = Path(config_path).resolve()
    if config_file.parent.name == "configs":
        project_root = config_file.parent.parent
    else:
        project_root = Path.cwd()

    # 작업 디렉토리를 프로젝트 루트로 변경 (runs/ 폴더 생성 방지)
    os.chdir(project_root)

    # 로그는 프로젝트 루트의 logs/ 폴더에 저장
    log_dir = project_root / "logs" / exp_name

    # 로거 설정
    logger = setup_logger(log_dir=log_dir)
    logger.info("Starting training...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Logs will be saved to: {log_dir}")

    # 시드 고정
    seed_val = exp_cfg.get('seed', 42)
    seed(seed_val)
    logger.info(f"Seed fixed to {seed_val}")

    # 디바이스 설정
    device = get_device(verbose=False)
    logger.info(f"Using device: {device}")

    # GPU 모델명 가져오기
    gpu_name = "CPU"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
        except Exception as e:
            logger.warning(f"Failed to get GPU name: {e}")
            gpu_name = "Unknown GPU"

    # 모델 생성
    logger.info(f"Creating model: {model_cfg['type']}{model_cfg['size']}")
    model = ModelFactory.create_model(
        model_type=model_cfg['type'],
        size=model_cfg['size'],
        pretrained=model_cfg.get('pretrained', True)
    )

    # W&B 설정 및 초기화
    is_yolo = model_cfg['type'].lower().startswith('yolo')

    # YOLO인 경우 settings를 먼저 설정 (wandb.init 전에)
    if is_yolo and wandb_cfg.get('enabled', False):
        try:
            from ultralytics import settings
            settings.update({'wandb': True})
            logger.info("Ultralytics wandb integration enabled via settings (before init)")
        except Exception as e:
            logger.warning(f"Failed to update ultralytics settings: {e}")

    if wandb_cfg.get('enabled', False):
        login_wandb()

        # run name 생성: 항상 {모델명}-{suffix} 형태
        wandb_suffix = wandb_cfg.get('suffix', exp_suffix)  # wandb.suffix 또는 experiment.suffix
        run_name = f"{model_name}-{wandb_suffix}"  # 예: 'yolo11n-base'

        # W&B config 깔끔하게 정리
        wandb_config = {
            # Model
            'model_type': model_cfg['type'],
            'model_size': model_cfg['size'],
            'pretrained': model_cfg.get('pretrained', True),

            # Training - Basic
            'epochs': train_cfg['epochs'],
            'batch_size': train_cfg['batch_size'],
            'img_size': data_cfg['img_size'],
            'patience': train_cfg.get('patience', 10),
            'device': str(device),
            'gpu_name': gpu_name,
            'workers': train_cfg.get('workers', 8),

            # Training - Optimizer
            'optimizer': train_cfg.get('optimizer', 'auto'),
            'lr0': train_cfg.get('lr0', 0.01),
            'lrf': train_cfg.get('lrf', 0.01),
            'momentum': train_cfg.get('momentum', 0.937),
            'weight_decay': train_cfg.get('weight_decay', 0.0005),
            'dropout': train_cfg.get('dropout', 0.0),

            # Augmentation
            'augmentation_enabled': config.get('augmentation', {}).get('enabled', False),
            'pcb_optimized': config.get('augmentation', {}).get('pcb_optimized', False),

            # Experiment
            'seed': exp_cfg.get('seed', 42),
            'run_name': exp_name,
        }

        if is_yolo:
            # YOLO는 자체 wandb 통합을 사용
            # wandb.init()을 먼저 호출하면 YOLO가 활성화된 run을 감지하고 자동으로 로깅
            wandb.init(
                project=wandb_cfg.get('project', 'PCB_Detection'),
                name=run_name,
                tags=wandb_cfg.get('tags', []),
                config=wandb_config
            )
            logger.info(f"W&B initialized for YOLO: project='{wandb_cfg.get('project')}', name='{run_name}'")
        else:
            # 다른 모델은 수동으로 wandb 초기화
            wandb.init(
                project=wandb_cfg.get('project', 'PCB_Detection'),
                name=run_name,
                tags=wandb_cfg.get('tags', []),
                config=wandb_config
            )
            logger.info(f"W&B initialized for project='{wandb_cfg.get('project')}', name='{run_name}'")

    # 학습 파라미터 설정
    # data_yaml을 절대 경로로 변환
    data_yaml_path = Path(data_cfg['data_yaml'])
    if not data_yaml_path.is_absolute():
        data_yaml_path = (project_root / data_yaml_path).resolve()

    # project_dir을 절대 경로로 변환
    output_dir = Path(project_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    train_params = {
        'data': str(data_yaml_path),
        'epochs': train_cfg['epochs'],
        'batch': train_cfg['batch_size'],
        'imgsz': data_cfg['img_size'],
        'patience': train_cfg.get('patience', 10),
        'device': device,
        'workers': train_cfg.get('workers', 8),
        'save': True,
        'save_period': train_cfg.get('save_period') or -1,  # None이면 -1 (저장 안함)
        'project': str(output_dir),
        'name': exp_name,
        'val': True,  # Validation 수행
        'plots': False,  # 로컬 이미지 파일 생성 안 함
        'verbose': False,  # 상세 출력 끄기
    }

    # 커스텀 wandb 콜백 추가 (YOLO 자동 통합이 작동 안 할 경우를 대비)
    if is_yolo and wandb_cfg.get('enabled', False):
        def on_fit_epoch_end(trainer):
            """Epoch 끝날 때마다 loss와 metrics를 wandb에 로깅"""
            if wandb.run is None:
                return

            metrics = {'epoch': trainer.epoch}

            # Train loss 계산
            train_loss = None
            if hasattr(trainer, 'label_loss_items') and hasattr(trainer, 'tloss'):
                loss_items = trainer.label_loss_items(trainer.tloss, prefix='train')
                metrics.update(loss_items)
                # 전체 train loss (box + cls + dfl)
                train_loss = sum([v for k, v in loss_items.items() if 'train/' in k])

            # Learning rate 추가
            if hasattr(trainer, 'lr') and trainer.lr:
                metrics.update(trainer.lr)

            # Validation metrics 추가
            val_loss = None
            mAP50 = None
            if hasattr(trainer, 'metrics') and trainer.metrics:
                metrics.update(trainer.metrics)
                # Validation loss 찾기
                if 'val/box_loss' in trainer.metrics:
                    val_loss = (trainer.metrics.get('val/box_loss', 0) +
                               trainer.metrics.get('val/cls_loss', 0) +
                               trainer.metrics.get('val/dfl_loss', 0))
                # mAP50 찾기
                mAP50 = trainer.metrics.get('metrics/mAP50(B)', None)
                if mAP50 is None:
                    mAP50 = trainer.metrics.get('metrics/mAP50', None)

            # Wandb에 로깅
            wandb.log(metrics, step=trainer.epoch)

            # 콘솔에 간단한 출력
            epoch_str = f"Epoch {trainer.epoch + 1}/{trainer.epochs}"
            output_parts = []
            if train_loss is not None:
                output_parts.append(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                output_parts.append(f"Val Loss: {val_loss:.4f}")
            if mAP50 is not None:
                output_parts.append(f"mAP50: {mAP50:.4f}")

            if output_parts:
                print(f"{epoch_str} - {', '.join(output_parts)}")

        # 콜백 등록
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
        logger.info("Custom W&B callback registered for loss logging")

    # 주석 처리된 augmentation 및 optimizer 설정은 더 이상 여기서 읽지 않습니다.
    # 해당 파라미터가 train_params에 전달되지 않으면 YOLOv8의 기본값이 사용됩니다.

    logger.info(f"Training parameters: {train_params}")

    # 학습 시작
    logger.info("Starting model training...")
    try:
        results = model.train(**train_params)
        logger.info("Training completed successfully")
        logger.info(f"Best model saved at: {train_params['project']}/{train_params['name']}/weights/best.pt")

        # W&B에 best 메트릭 기록
        if wandb_cfg.get('enabled', False) and wandb.run is not None:
            try:
                # Validation 결과에서 best 메트릭 추출
                best_metrics = {
                    'best/mAP50': float(results.box.map50) if hasattr(results, 'box') else 0.0,
                    'best/mAP50-95': float(results.box.map) if hasattr(results, 'box') else 0.0,
                    'best/precision': float(results.box.mp) if hasattr(results, 'box') else 0.0,
                    'best/recall': float(results.box.mr) if hasattr(results, 'box') else 0.0,
                    'best/fitness': float(results.fitness) if hasattr(results, 'fitness') else 0.0,
                }

                # Early stopping 정보 추출
                trainer = model.trainer
                stopped_epoch = trainer.epoch + 1  # 실제 완료된 epoch 수
                total_epochs = train_cfg['epochs']
                early_stopped = stopped_epoch < total_epochs  # 설정된 epoch보다 일찍 멈췄으면 True

                # Best epoch 찾기 (results.csv에서 가장 높은 mAP50을 기록한 epoch)
                best_epoch = trainer.best_epoch if hasattr(trainer, 'best_epoch') else stopped_epoch

                # W&B summary에 저장 (run 비교 시 표시됨)
                for key, value in best_metrics.items():
                    wandb.run.summary[key] = value

                # Early stopping 정보 추가
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.summary['early_stopped'] = early_stopped
                wandb.run.summary['stopped_epoch'] = stopped_epoch

                logger.info(f"Best metrics logged to W&B: mAP50={best_metrics['best/mAP50']:.4f}, "
                           f"mAP50-95={best_metrics['best/mAP50-95']:.4f}")
                logger.info(f"Early stopping info: best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, early_stopped={early_stopped}")
            except Exception as e:
                logger.warning(f"Failed to log best metrics to W&B: {e}")

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
