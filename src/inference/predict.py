import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from ultralytics import YOLO
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """PCB 결함 검출 예측 클래스"""

    def __init__(self, weights_path: str, conf: float = 0.25, iou: float = 0.45):
        """
        Args:
            weights_path: 학습된 모델 가중치 경로
            conf: confidence threshold (기본값: 0.25)
            iou: IoU threshold for NMS (기본값: 0.45)
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        self.model = YOLO(str(self.weights_path))
        self.conf = conf
        self.iou = iou
        logger.info(f"Loaded model from: {weights_path}")
        logger.info(f"Confidence threshold: {conf}, IoU threshold: {iou}")

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        save: bool = True,
        save_dir: Optional[str] = None,
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: int = 2
    ) -> List[Dict[str, Any]]:
        """
        이미지 또는 이미지 폴더에 대해 예측 수행

        Args:
            source: 이미지 경로, 폴더 경로, 또는 numpy 배열
            save: 결과 이미지 저장 여부
            save_dir: 결과 저장 디렉토리 (None이면 자동 생성)
            show_labels: 클래스 라벨 표시 여부
            show_conf: confidence 점수 표시 여부
            line_width: bounding box 선 두께

        Returns:
            예측 결과 리스트 (각 이미지별 결과)
        """
        logger.info(f"Running prediction on: {source}")

        results = self.model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            save=save,
            project=save_dir or "runs/predict",
            name="exp",
            exist_ok=True,
            show_labels=show_labels,
            show_conf=show_conf,
            line_width=line_width
        )

        # 결과 파싱
        predictions = []
        for result in results:
            pred_dict = {
                'image_path': result.path,
                'image_shape': result.orig_shape,
                'boxes': [],
                'num_detections': len(result.boxes)
            }

            # 각 detection 정보 추출
            for box in result.boxes:
                box_dict = {
                    'xyxy': box.xyxy.cpu().numpy().tolist()[0],  # [x1, y1, x2, y2]
                    'conf': float(box.conf.cpu().numpy()[0]),
                    'cls': int(box.cls.cpu().numpy()[0]),
                    'class_name': result.names[int(box.cls.cpu().numpy()[0])]
                }
                pred_dict['boxes'].append(box_dict)

            predictions.append(pred_dict)
            logger.info(f"Detected {pred_dict['num_detections']} objects in {result.path}")

        return predictions

    def predict_single(
        self,
        image_path: Union[str, Path],
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        단일 이미지에 대한 예측

        Args:
            image_path: 이미지 경로
            visualize: 결과 시각화 여부

        Returns:
            예측 결과 딕셔너리
        """
        predictions = self.predict(source=image_path, save=visualize)
        return predictions[0] if predictions else None

    def predict_batch(
        self,
        image_dir: Union[str, Path],
        pattern: str = "*.jpg",
        save: bool = True,
        save_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        폴더 내 모든 이미지에 대한 배치 예측

        Args:
            image_dir: 이미지 폴더 경로
            pattern: 이미지 파일 패턴 (기본값: "*.jpg")
            save: 결과 저장 여부
            save_dir: 결과 저장 디렉토리

        Returns:
            전체 예측 결과 리스트
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_files = list(image_dir.glob(pattern))
        if not image_files:
            logger.warning(f"No images found with pattern '{pattern}' in {image_dir}")
            return []

        logger.info(f"Found {len(image_files)} images in {image_dir}")
        return self.predict(source=str(image_dir), save=save, save_dir=save_dir)


def predict_from_weights(
    weights_path: str,
    source: Union[str, Path],
    conf: float = 0.25,
    iou: float = 0.45,
    save: bool = True,
    save_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    편의 함수: 가중치에서 직접 예측 수행

    Args:
        weights_path: 모델 가중치 경로
        source: 이미지 경로 또는 폴더
        conf: confidence threshold
        iou: IoU threshold
        save: 결과 저장 여부
        save_dir: 결과 저장 디렉토리

    Returns:
        예측 결과 리스트
    """
    predictor = Predictor(weights_path, conf=conf, iou=iou)
    return predictor.predict(source=source, save=save, save_dir=save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCB Defect Detection Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true", help="Save prediction results")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 예측 실행
    predictor = PCBPredictor(args.weights, conf=args.conf, iou=args.iou)
    results = predictor.predict(
        source=args.source,
        save=args.save,
        save_dir=args.save_dir
    )

    logger.info(f"Prediction completed. Total detections: {sum(r['num_detections'] for r in results)}")
