import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import albumentations as A
import cv2

from src.utils.config import load_config
from src.utils.utils import setup_logger, get_device

logger = logging.getLogger(__name__)

# 클래스 이름 매핑
CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}


def get_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('augmentation', {})


def get_train_transforms(
    img_size: int = 640,
    augmentation_config: Optional[Dict[str, Any]] = None,
    pcb_optimized: bool = True
) -> A.Compose:
    """
    학습용 augmentation 파이프라인 생성
    PCB 데이터에 최적화: 밝기/그림자/회전/반전 중심

    Args:
        img_size: 이미지 크기
        augmentation_config: augmentation 설정 딕셔너리
        pcb_optimized: PCB 데이터에 최적화된 설정 사용 여부

    Returns:
        albumentations Compose 객체
    """
    if augmentation_config is None:
        augmentation_config = {}

    logger.info(f"Creating train transforms with img_size={img_size}")
    logger.info(f"PCB optimized mode: {pcb_optimized}")
    logger.info(f"Augmentation config: {augmentation_config}")

    if pcb_optimized:
        # PCB 특성에 맞는 augmentation
        transforms = [
            # 1. 밝기/대비 조정 (가장 중요 - PCB 조명 환경 시뮬레이션)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # 밝기 변화 강화
                contrast_limit=0.3,    # 대비 변화 강화
                p=0.7  # 높은 확률로 적용
            ),

            # 2. 그림자/조명 효과 (PCB 검사 환경)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),  # 그림자 영역
                num_shadows_limit=(1, 2),   # 그림자 개수
                shadow_dimension=5,
                p=0.4
            ),

            # 3. 회전 (PCB 방향)
            A.Rotate(
                limit=augmentation_config.get('degrees', 10),  # 기본 10도
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),

            # 4. 좌우 반전 (PCB 좌우 대칭)
            A.HorizontalFlip(
                p=augmentation_config.get('fliplr', 0.5)
            ),

            # 5. 상하 반전 (필요시)
            A.VerticalFlip(
                p=augmentation_config.get('flipud', 0.3)
            ),

            # 6. 이동/스케일 (약간만)
            A.ShiftScaleRotate(
                shift_limit=augmentation_config.get('translate', 0.1),
                scale_limit=augmentation_config.get('scale', 0.1),  # 스케일 변화 줄임
                rotate_limit=0,  # 회전은 위에서 처리
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),

            # 7. 노이즈 (미세한 센서 노이즈)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),  # 노이즈 줄임
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.3),

            # 8. 블러 (초점 문제 시뮬레이션 - 최소화)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.2),  # 확률 낮춤

            # 9. 색상 조정 (최소화 - PCB는 주로 녹색/갈색)
            A.HueSaturationValue(
                hue_shift_limit=5,   # 색상 변화 최소화
                sat_shift_limit=10,  # 채도 변화 최소화
                val_shift_limit=10,  # 명도 변화 최소화
                p=0.2  # 낮은 확률
            ),

            # 10. 리사이즈 (항상 적용)
            A.Resize(height=img_size, width=img_size, p=1.0),
        ]
    else:
        # 기존 범용 augmentation
        hsv_h = augmentation_config.get('hsv_h', 0.015)
        hsv_s = augmentation_config.get('hsv_s', 0.7)
        hsv_v = augmentation_config.get('hsv_v', 0.4)
        degrees = augmentation_config.get('degrees', 0.0)
        translate = augmentation_config.get('translate', 0.1)
        scale = augmentation_config.get('scale', 0.5)
        flipud = augmentation_config.get('flipud', 0.0)
        fliplr = augmentation_config.get('fliplr', 0.5)

        transforms = [
            A.HueSaturationValue(
                hue_shift_limit=int(hsv_h * 180),
                sat_shift_limit=int(hsv_s * 100),
                val_shift_limit=int(hsv_v * 100),
                p=0.5
            ),
            A.Affine(
                rotate=(-degrees, degrees) if degrees > 0 else 0,
                translate_percent={
                    'x': (-translate, translate),
                    'y': (-translate, translate)
                } if translate > 0 else None,
                scale=(1 - scale, 1 + scale) if scale > 0 else None,
                p=0.5
            ),
            A.VerticalFlip(p=flipud),
            A.HorizontalFlip(p=fliplr),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.Resize(height=img_size, width=img_size, p=1.0),
        ]

    # bbox_params 설정 (YOLO 형식)
    bbox_params = A.BboxParams(
        format='yolo',
        min_area=0,
        min_visibility=0.3,
        label_fields=['class_labels']
    )

    return A.Compose(transforms, bbox_params=bbox_params)


def get_val_transforms(img_size: int = 640) -> A.Compose:
    logger.info(f"Creating validation transforms with img_size={img_size}")

    transforms = [
        # 리사이즈만 수행
        A.Resize(height=img_size, width=img_size, p=1.0),
    ]

    # bbox_params 설정 (YOLO 형식)
    bbox_params = A.BboxParams(
        format='yolo',
        min_area=0,
        min_visibility=0.0,
        label_fields=['class_labels']
    )

    return A.Compose(transforms, bbox_params=bbox_params)


class AugmentationPipeline:
    """
    Data augmentation 파이프라인 클래스
    train_config.yaml의 augmentation 설정에 따라 동적으로 증강 적용
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        img_size: int = 640,
        mode: str = 'train',
        log_dir: Optional[str] = None
    ):
        """
        Args:
            config_path: 설정 파일 경로
            config: 설정 딕셔너리 (config_path와 중복 시 이것을 우선 사용)
            img_size: 이미지 크기
            mode: 'train' 또는 'val'
            log_dir: 로그 디렉토리
        """
        self.img_size = img_size
        self.mode = mode

        # 로거 설정
        self.logger = setup_logger(
            name=f"Augmentation_{mode}",
            log_dir=log_dir
        )

        # 디바이스 설정 (필요시 사용)
        self.device = get_device(verbose=False)
        self.logger.info(f"Using device: {self.device}")

        # 설정 로드
        if config is None and config_path is not None:
            config = load_config(config_path)
            self.logger.info(f"Loaded config from {config_path}")
        elif config is None:
            config = {}
            self.logger.warning("No config provided, using default settings")

        self.config = config
        self.augmentation_config = get_augmentation_config(config)

        # 증강 설정 읽기
        self.enabled = self.augmentation_config.get('enabled', True)
        self.pcb_optimized = self.augmentation_config.get('pcb_optimized', True)
        self.exclude_classes = self._process_exclude_classes(
            self.augmentation_config.get('exclude_classes', [])
        )

        self.logger.info(f"Augmentation enabled: {self.enabled}")
        self.logger.info(f"PCB optimized: {self.pcb_optimized}")
        if self.exclude_classes:
            exclude_names = [CLASS_NAMES[cls_id] for cls_id in self.exclude_classes]
            self.logger.info(f"Exclude classes: {exclude_names} (IDs: {list(self.exclude_classes)})")

        # Transform 생성
        if mode == 'train' and self.enabled:
            # 학습 모드 + 증강 활성화
            self.transform = get_train_transforms(
                img_size=img_size,
                augmentation_config=self.augmentation_config,
                pcb_optimized=self.pcb_optimized
            )
            # 제외 클래스용 최소 변환 (리사이즈만)
            self.minimal_transform = get_val_transforms(img_size=img_size)
            self.logger.info("Created training augmentation pipeline")
        else:
            # 검증 모드 또는 증강 비활성화 → 리사이즈만
            self.transform = get_val_transforms(img_size=img_size)
            self.minimal_transform = self.transform
            if mode == 'train' and not self.enabled:
                self.logger.info("Augmentation disabled - using minimal transforms only")
            else:
                self.logger.info("Created validation augmentation pipeline")

    def _process_exclude_classes(
        self,
        exclude_classes: Optional[Union[List[str], List[int]]]
    ) -> set:
        """
        제외할 클래스를 클래스 ID set으로 변환

        Args:
            exclude_classes: 클래스 이름 또는 ID 리스트

        Returns:
            클래스 ID set
        """
        if not exclude_classes:
            return set()

        exclude_ids = set()
        for cls in exclude_classes:
            if isinstance(cls, int):
                # 클래스 ID인 경우
                if cls in CLASS_NAMES:
                    exclude_ids.add(cls)
                else:
                    self.logger.warning(f"Unknown class ID: {cls}")
            elif isinstance(cls, str):
                # 클래스 이름인 경우
                found = False
                for cls_id, cls_name in CLASS_NAMES.items():
                    if cls_name == cls:
                        exclude_ids.add(cls_id)
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Unknown class name: {cls}")
            else:
                self.logger.warning(f"Invalid class type: {type(cls)}")

        return exclude_ids

    def __call__(
        self,
        image,
        bboxes: Optional[List] = None,
        class_labels: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        이미지와 bbox에 augmentation 적용
        exclude_classes에 포함된 클래스는 최소 변환만 적용

        Args:
            image: 입력 이미지 (numpy array or path)
            bboxes: bounding boxes (YOLO 형식: [x_center, y_center, width, height])
            class_labels: 클래스 레이블

        Returns:
            변환된 이미지와 bbox를 포함한 딕셔너리
        """
        # 이미지 로드 (경로인 경우)
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # bbox가 없는 경우 (classification 등)
        if bboxes is None:
            transformed = self.transform(image=image)
            return transformed

        # bbox가 있는 경우 (object detection)
        if class_labels is None:
            class_labels = [0] * len(bboxes)

        # exclude_classes 체크: 이미지에 제외 클래스만 있는 경우 최소 변환
        if self.exclude_classes and self.mode == 'train':
            image_classes = set(class_labels)
            # 모든 클래스가 제외 대상인 경우
            if image_classes.issubset(self.exclude_classes):
                try:
                    transformed = self.minimal_transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    return transformed
                except Exception as e:
                    self.logger.error(f"Minimal transform failed: {e}")
                    return {
                        'image': image,
                        'bboxes': bboxes,
                        'class_labels': class_labels
                    }

        # 일반 augmentation 적용
        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed
        except Exception as e:
            self.logger.error(f"Augmentation failed: {e}")
            # 실패 시 원본 반환
            return {
                'image': image,
                'bboxes': bboxes,
                'class_labels': class_labels
            }

    def get_yolo_format_params(self) -> Dict[str, Any]:
        if not self.augmentation_config:
            self.logger.info("No augmentation config, returning empty params")
            return {}

        # YOLOv8/v11이 지원하는 파라미터로 변환
        yolo_params = {}

        # HSV 증강
        if 'hsv_h' in self.augmentation_config:
            yolo_params['hsv_h'] = self.augmentation_config['hsv_h']
        if 'hsv_s' in self.augmentation_config:
            yolo_params['hsv_s'] = self.augmentation_config['hsv_s']
        if 'hsv_v' in self.augmentation_config:
            yolo_params['hsv_v'] = self.augmentation_config['hsv_v']

        # 기하학적 변환
        if 'degrees' in self.augmentation_config:
            yolo_params['degrees'] = self.augmentation_config['degrees']
        if 'translate' in self.augmentation_config:
            yolo_params['translate'] = self.augmentation_config['translate']
        if 'scale' in self.augmentation_config:
            yolo_params['scale'] = self.augmentation_config['scale']

        # 반전
        if 'flipud' in self.augmentation_config:
            yolo_params['flipud'] = self.augmentation_config['flipud']
        if 'fliplr' in self.augmentation_config:
            yolo_params['fliplr'] = self.augmentation_config['fliplr']

        # Mosaic (YOLOv8 특화)
        if 'mosaic' in self.augmentation_config:
            yolo_params['mosaic'] = self.augmentation_config['mosaic']

        self.logger.info(f"YOLO augmentation params: {yolo_params}")
        return yolo_params


def create_augmentation_pipeline(
    config_path: str = "configs/train_config.yaml",
    img_size: int = 640,
    mode: str = 'train',
    log_dir: Optional[str] = None
) -> AugmentationPipeline:
    return AugmentationPipeline(
        config_path=config_path,
        img_size=img_size,
        mode=mode,
        log_dir=log_dir
    )


if __name__ == "__main__":
    """
    사용 예시
    """
    import numpy as np

    # 설정 로드 및 파이프라인 생성
    pipeline = create_augmentation_pipeline(
        config_path="configs/train_config.yaml",
        img_size=640,
        mode='train'
    )

    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_bboxes = [[0.5, 0.5, 0.2, 0.2]]  # [x_center, y_center, width, height]
    test_labels = [0]

    # Augmentation 적용
    result = pipeline(
        image=test_image,
        bboxes=test_bboxes,
        class_labels=test_labels
    )

    print("Augmentation result:")
    print(f"Image shape: {result['image'].shape}")
    print(f"Bboxes: {result['bboxes']}")
    print(f"Labels: {result['class_labels']}")

    # YOLO 형식 파라미터 출력
    yolo_params = pipeline.get_yolo_format_params()
    print(f"\nYOLO augmentation params: {yolo_params}")
