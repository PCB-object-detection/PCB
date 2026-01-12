import logging
from ultralytics import YOLO
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_model_info(model: YOLO) -> Dict[str, Any]:
    try:
        info = {
            'model_type': model.model_name if hasattr(model, 'model_name') else 'Unknown',
            'task': model.task,
        }

        # 파라미터 수 계산
        if hasattr(model, 'model'):
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(
                p.numel() for p in model.model.parameters() if p.requires_grad
            )
            info['total_params'] = f"{total_params:,}"
            info['trainable_params'] = f"{trainable_params:,}"

        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {'error': str(e)}


def save_model(model: YOLO, save_path: str) -> None:
    try:
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def load_model_weights(model: YOLO, weights_path: str) -> YOLO:
    try:
        model = YOLO(weights_path)
        logger.info(f"Loaded weights from {weights_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise
