import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ultralytics import YOLO
import json
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class Evaluator:
    """PCB 결함 검출 모델 평가 클래스"""

    def __init__(self, weights_path: str):
        """
        Args:
            weights_path: 학습된 모델 가중치 경로
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        self.model = YOLO(str(self.weights_path))
        logger.info(f"Loaded model from: {weights_path}")

    def evaluate(
        self,
        data_yaml: str,
        split: str = "test",
        conf: float = 0.001,
        iou: float = 0.6,
        save_dir: Optional[str] = None,
        plots: bool = False
    ) -> Dict[str, Any]:
        """
        모델 성능 평가

        Args:
            data_yaml: 데이터셋 yaml 파일 경로
            split: 평가할 데이터셋 분할 ('train', 'val', 'test')
            conf: confidence threshold (기본값: 0.001, 낮게 설정하여 모든 예측 포함)
            iou: IoU threshold for mAP calculation (기본값: 0.6)
            save_dir: 결과 저장 디렉토리
            plots: 그래프 생성 여부

        Returns:
            평가 메트릭 딕셔너리
        """
        logger.info(f"Starting evaluation on {split} set...")
        logger.info(f"Data config: {data_yaml}")
        logger.info(f"Conf threshold: {conf}, IoU threshold: {iou}")

        # Ultralytics val() 메서드 사용
        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=conf,
            iou=iou,
            save=False,  # 결과 저장 안 함
            plots=plots,
            save_json=False,  # JSON 저장 안 함
            verbose=False  # 상세 출력 끄기
        )

        # 메트릭 추출
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'fitness': float(results.fitness),
        }

        # 클래스별 메트릭
        if hasattr(results.box, 'maps'):
            class_metrics = {}
            for i, class_map in enumerate(results.box.maps):
                class_name = results.names[i] if hasattr(results, 'names') else f"class_{i}"
                class_metrics[class_name] = {
                    'mAP50-95': float(class_map),
                    'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else None,
                    'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else None,
                }
            metrics['per_class'] = class_metrics

        logger.info(f"Evaluation completed!")
        logger.info(f"mAP50: {metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")

        return metrics

    def evaluate_and_save(
        self,
        data_yaml: str,
        split: str = "test",
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        평가 수행 후 결과를 JSON 파일로 저장

        Args:
            data_yaml: 데이터셋 yaml 파일 경로
            split: 평가할 데이터셋 분할
            output_path: JSON 결과 파일 경로
            **kwargs: evaluate() 함수에 전달할 추가 인자

        Returns:
            평가 메트릭 딕셔너리
        """
        metrics = self.evaluate(data_yaml=data_yaml, split=split, **kwargs)

        # 결과 저장
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Evaluation results saved to: {output_path}")

        return metrics

    def compare_models(
        self,
        other_weights: Union[str, list],
        data_yaml: str,
        split: str = "test",
        save_comparison: bool = True,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        여러 모델 성능 비교

        Args:
            other_weights: 비교할 다른 모델 가중치 경로 (단일 또는 리스트)
            data_yaml: 데이터셋 yaml 파일 경로
            split: 평가할 데이터셋 분할
            save_comparison: 비교 결과 저장 여부
            output_path: 비교 결과 CSV 파일 경로

        Returns:
            비교 결과 DataFrame
        """
        if isinstance(other_weights, str):
            other_weights = [other_weights]

        all_weights = [str(self.weights_path)] + other_weights
        results_list = []

        for weights in all_weights:
            logger.info(f"Evaluating model: {weights}")
            evaluator = Evaluator(weights)
            metrics = evaluator.evaluate(data_yaml=data_yaml, split=split, plots=False)

            result = {
                'model': Path(weights).stem,
                'weights_path': weights,
                **{k: v for k, v in metrics.items() if k != 'per_class'}
            }
            results_list.append(result)

        # DataFrame 생성
        df = pd.DataFrame(results_list)
        df = df.sort_values('mAP50-95', ascending=False)

        logger.info("\n" + "="*50)
        logger.info("Model Comparison Results:")
        logger.info("="*50)
        logger.info(f"\n{df.to_string()}")

        # 결과 저장
        if save_comparison and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Comparison results saved to: {output_path}")

        return df


def evaluate_model(
    weights_path: str,
    data_yaml: str,
    split: str = "test",
    conf: float = 0.001,
    iou: float = 0.6,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    편의 함수: 모델 평가

    Args:
        weights_path: 모델 가중치 경로
        data_yaml: 데이터셋 yaml 파일 경로
        split: 평가할 데이터셋 분할
        conf: confidence threshold
        iou: IoU threshold
        save_dir: 결과 저장 디렉토리

    Returns:
        평가 메트릭 딕셔너리
    """
    evaluator = Evaluator(weights_path)
    return evaluator.evaluate(
        data_yaml=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        save_dir=save_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCB Defect Detection Model Evaluation")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for mAP")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 평가 실행
    evaluator = Evaluator(args.weights)
    metrics = evaluator.evaluate_and_save(
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_dir=args.save_dir,
        output_path=args.output
    )

    logger.info("Evaluation completed successfully!")
