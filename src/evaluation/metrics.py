import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    두 박스 간의 IoU (Intersection over Union) 계산

    Args:
        box1: [x1, y1, x2, y2] 형식의 박스
        box2: [x1, y1, x2, y2] 형식의 박스

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 면적
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 박스 면적
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합 면적
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def calculate_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> Tuple[float, float, int, int, int]:
    """
    Precision과 Recall 계산

    Args:
        predictions: 예측 결과 리스트 (각 요소는 {'boxes': [...], 'classes': [...], 'confidences': [...]})
        ground_truths: Ground truth 리스트
        iou_threshold: IoU 임계값
        conf_threshold: Confidence 임계값

    Returns:
        (precision, recall, TP, FP, FN)
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = [box for box, conf in zip(pred['boxes'], pred['confidences'])
                      if conf >= conf_threshold]
        gt_boxes = gt['boxes']

        matched_gt = set()

        # 각 예측 박스에 대해
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            # 가장 잘 매칭되는 GT 박스 찾기
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # IoU 임계값을 넘으면 TP, 아니면 FP
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        # 매칭되지 않은 GT는 FN
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall, tp, fp, fn


def calculate_f1_score(precision: float, recall: float) -> float:
    """F1 Score 계산"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
    method: str = '11point'
) -> float:
    """
    Average Precision (AP) 계산

    Args:
        precisions: Precision 배열
        recalls: Recall 배열
        method: 계산 방법 ('11point' 또는 'interp')

    Returns:
        AP 값
    """
    if method == '11point':
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t]
            ap += np.max(p) if len(p) > 0 else 0.0
        return ap / 11.0

    elif method == 'interp':
        # All-point interpolation (COCO style)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Mean Average Precision (mAP) 계산

    Args:
        predictions: 예측 결과 리스트
        ground_truths: Ground truth 리스트
        num_classes: 클래스 개수
        iou_threshold: IoU 임계값

    Returns:
        클래스별 AP와 mAP를 포함한 딕셔너리
    """
    class_aps = {}

    for class_id in range(num_classes):
        # 해당 클래스의 예측과 GT만 필터링
        class_preds = []
        class_gts = []

        for pred, gt in zip(predictions, ground_truths):
            pred_class_boxes = [box for box, cls in zip(pred['boxes'], pred['classes'])
                                if cls == class_id]
            gt_class_boxes = [box for box, cls in zip(gt['boxes'], gt['classes'])
                              if cls == class_id]

            class_preds.append({'boxes': pred_class_boxes,
                                'confidences': [c for c, cls in zip(pred['confidences'], pred['classes'])
                                                if cls == class_id]})
            class_gts.append({'boxes': gt_class_boxes})

        # Precision-Recall 계산
        # (여기서는 간단히 전체에 대한 precision/recall만 계산)
        precision, recall, _, _, _ = calculate_precision_recall(
            class_preds, class_gts, iou_threshold
        )

        class_aps[f'class_{class_id}'] = precision * recall  # 간단한 근사

    # mAP 계산
    map_value = np.mean(list(class_aps.values())) if class_aps else 0.0

    return {
        'mAP': map_value,
        **class_aps
    }


def calculate_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Confusion Matrix 계산

    Args:
        predictions: 예측 결과 리스트
        ground_truths: Ground truth 리스트
        num_classes: 클래스 개수 (배경 포함하여 +1)
        iou_threshold: IoU 임계값

    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    # 배경 클래스를 포함하여 confusion matrix 생성
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_classes = pred['classes']
        gt_boxes = gt['boxes']
        gt_classes = gt['classes']

        matched_gt = set()

        # 각 예측에 대해
        for pred_box, pred_cls in zip(pred_boxes, pred_classes):
            best_iou = 0
            best_gt_idx = -1
            best_gt_cls = -1

            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_cls = gt_cls

            if best_iou >= iou_threshold:
                # 매칭된 경우
                conf_matrix[best_gt_cls, pred_cls] += 1
                matched_gt.add(best_gt_idx)
            else:
                # 배경으로 분류 (FP)
                conf_matrix[num_classes, pred_cls] += 1

        # 매칭되지 않은 GT (FN)
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                conf_matrix[gt_cls, num_classes] += 1

    return conf_matrix


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """메트릭 결과를 보기 좋게 출력"""
    logger.info("\n" + "="*50)
    logger.info("Evaluation Metrics Summary")
    logger.info("="*50)

    for key, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value:.4f}")
        else:
            logger.info(f"{key}: {value:.4f}")

    logger.info("="*50)


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)

    # 예제 데이터
    pred_box1 = [10, 10, 50, 50]
    gt_box1 = [15, 15, 55, 55]

    iou = calculate_iou(pred_box1, gt_box1)
    logger.info(f"IoU between boxes: {iou:.4f}")

    # Precision/Recall 테스트
    predictions = [
        {
            'boxes': [[10, 10, 50, 50], [60, 60, 100, 100]],
            'classes': [0, 1],
            'confidences': [0.9, 0.8]
        }
    ]
    ground_truths = [
        {
            'boxes': [[15, 15, 55, 55], [65, 65, 105, 105]],
            'classes': [0, 1]
        }
    ]

    precision, recall, tp, fp, fn = calculate_precision_recall(
        predictions, ground_truths, iou_threshold=0.5
    )
    f1 = calculate_f1_score(precision, recall)

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}")
