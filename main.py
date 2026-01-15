import argparse
import logging
from pathlib import Path
from src.utils.config import load_config, get_data_config
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
CONFIG_PATH = 'configs/config.yaml'

def get_best_weights_path(config):
    """config에서 best model 경로 추출"""
    project_dir = Path(config['training']['save_dir'])
    model_type = config['model']['type']
    model_size = config['model']['size']
    exp_suffix = config['experiment']['suffix']
    exp_name = f"{model_type}{model_size}-{exp_suffix}"
    return project_dir / exp_name / 'weights' / 'best.pt'


def run_pipeline():
    """전체 파이프라인"""
    logger.info("Starting training...")
    train_result = train_model(CONFIG_PATH)

    if 'model' not in train_result:
        logger.error("Training failed!")
        return None

    logger.info("Starting test evaluation...")
    config = train_result['config']
    data_cfg = get_data_config(config)
    best_weights = get_best_weights_path(config)

    if not best_weights.exists():
        logger.warning(f"Best weights not found: {best_weights}")
        return train_result

    eval_metrics = evaluate_model(
        weights_path=str(best_weights),
        data_yaml=data_cfg['data_yaml'],
        split='test'
    )

    logger.info(f"Test mAP50: {eval_metrics['mAP50']:.4f}, mAP50-95: {eval_metrics['mAP50-95']:.4f}")

    return {
        'training': train_result,
        'evaluation': eval_metrics
    }


def run_train():
    """학습만 실행"""
    return train_model(CONFIG_PATH)


def run_evaluation(weights_path=None):
    """평가만 실행 (weights_path=None이면 자동 탐색)"""
    config = load_config(CONFIG_PATH)
    data_cfg = get_data_config(config)

    if weights_path:
        weights = Path(weights_path)
        if not weights.exists():
            logger.error(f"Weights not found: {weights}")
            return None
        logger.info(f"Using specified weights: {weights}")
    else:
        weights = get_best_weights_path(config)
        if not weights.exists():
            logger.error(f"Best weights not found: {weights}")
            logger.error("Train a model first or specify weights with --weights")
            return None
        logger.info(f"Using best weights: {weights}")

    eval_metrics = evaluate_model(
        weights_path=str(weights),
        data_yaml=data_cfg['data_yaml'],
        split='test'
    )

    logger.info(f"Test mAP50: {eval_metrics['mAP50']:.4f}, mAP50-95: {eval_metrics['mAP50-95']:.4f}")
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description='PCB Defect Detection Pipeline')
    parser.add_argument(
        '--mode',
        default='pipeline',
        choices=['train', 'eval', 'pipeline'],
        help='train | eval | pipeline (default)'
    )
    parser.add_argument(
        '--weights',
        default=None,
        help='Model weights path for eval mode (default: auto-detect)'
    )
    args = parser.parse_args()

    if args.mode == 'train':
        return run_train()
    elif args.mode == 'eval':
        return run_evaluation(weights_path=args.weights)
    else:
        return run_pipeline()


if __name__ == "__main__":
    main()
