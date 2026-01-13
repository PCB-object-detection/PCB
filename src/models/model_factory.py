import logging
import warnings
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any
from src.models.model import get_model_info

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelFactory:
    # 지원하는 모델 설정
    SUPPORTED_MODELS = {
        'yolov5': ['n', 's', 'm', 'l'],
        'yolov8': ['n', 's', 'm', 'l'],
        'yolov11': ['n', 's', 'm', 'l'],
        'yolo11': ['n', 's', 'm', 'l'],  # v 없는 버전
    }

    @classmethod
    def create_model(
        cls,
        model_type: str = 'yolov8',
        size: str = 'n',
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ) -> YOLO:
        """
        모델 생성 (팩토리 메서드)

        Args:
            model_type: 모델 타입 ('yolov5', 'yolov8', 'yolov11')
            size: 모델 크기 ('n', 's', 'm', 'l')
            pretrained: 사전학습 가중치 사용 여부
            weights_path: 커스텀 가중치 파일 경로 (선택사항)

        Returns:
            YOLO 모델 객체

        Examples:
            >>> model = ModelFactory.create_model('yolov8', 'n')
            >>> model = ModelFactory.create_model('yolov5', 's', pretrained=True)
            >>> model = ModelFactory.create_model(weights_path='weights/best.pt')
        """
        # 커스텀 가중치가 있으면 우선 사용
        if weights_path:
            return cls._load_from_weights(weights_path)

        # 입력 검증
        cls._validate_inputs(model_type, size)

        # 모델 생성
        model_name = cls._get_model_name(model_type, size, pretrained)
        logger.info(f"Creating {model_type.upper()}{size.upper()} model...")
        logger.info(f"Loading model from: {model_name}")

        try:
            model = YOLO(model_name)
            logger.info(f"Successfully created {model_type.upper()}{size.upper()} model")
            return model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    @classmethod
    def _load_from_weights(cls, weights_path: str) -> YOLO:
        """가중치 파일에서 모델 로드"""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        logger.info(f"Loading model from weights: {weights_path}")
        try:
            model = YOLO(str(weights_path))
            logger.info(f"Successfully loaded model from {weights_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from weights: {e}")
            raise

    @classmethod
    def _validate_inputs(cls, model_type: str, size: str) -> None:
        """입력 검증"""
        model_type = model_type.lower()
        size = size.lower()

        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Choose from: {list(cls.SUPPORTED_MODELS.keys())}"
            )

        if size not in cls.SUPPORTED_MODELS[model_type]:
            raise ValueError(
                f"Unsupported size '{size}' for {model_type}. "
                f"Choose from: {cls.SUPPORTED_MODELS[model_type]}"
            )

    @classmethod
    def _download_to_weights(cls, model_type: str, size: str) -> None:
        """weights 폴더에 모델 다운로드"""
        import shutil
        import os
        import tempfile

        model_name = f"{model_type}{size}.pt"
        weights_dir = Path("weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / model_name

        logger.info(f"Downloading {model_name} to {weights_dir}/")

        # 임시 디렉토리에 다운로드 후 복사
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = os.getcwd()
            os.chdir(temp_dir)

            try:
                # YOLO가 임시 디렉토리에 다운로드
                logger.info(f"Downloading to temporary directory: {temp_dir}")
                temp_model = YOLO(model_name)

                # 다운로드된 파일을 weights 폴더로 복사
                temp_file = Path(temp_dir) / model_name
                if temp_file.exists():
                    shutil.copy2(temp_file, weights_path)
                    logger.info(f"Successfully downloaded and copied to {weights_path}")
                else:
                    raise FileNotFoundError(f"Downloaded file not found in {temp_dir}")
            finally:
                os.chdir(original_dir)

    @classmethod
    def _get_model_name(cls, model_type: str, size: str, pretrained: bool) -> str:
        """모델 이름 생성"""
        model_type = model_type.lower()
        size = size.lower()

        if pretrained:
            model_name = f"{model_type}{size}.pt"
            weights_path = Path("weights") / model_name

            # weights 폴더에서 파일 찾기 (v 있는 버전, 없는 버전 모두 시도)
            if weights_path.exists():
                logger.info(f"Found weights: {weights_path}")
                return str(weights_path)

            # yolov11 → yolo11 또는 yolo11 → yolov11 시도
            if model_type.startswith("yolov"):
                alt_model_type = model_type.replace("yolov", "yolo")
            else:
                alt_model_type = model_type.replace("yolo", "yolov")

            alt_model_name = f"{alt_model_type}{size}.pt"
            alt_weights_path = Path("weights") / alt_model_name

            if alt_weights_path.exists():
                logger.info(f"Found weights (alternative name): {alt_weights_path}")
                return str(alt_weights_path)

            # 둘 다 없으면 모델 이름만 반환 (YOLO가 자동 다운로드)
            logger.warning(f"Weights not found in weights/ folder: {model_name}")
            logger.info(f"Will download from Ultralytics hub automatically...")
            return model_name
        else:
            return f"{model_type}{size}.yaml"

    @classmethod
    def get_supported_models(cls) -> Dict[str, list]:
        """지원하는 모델 목록 반환"""
        return cls.SUPPORTED_MODELS.copy()


# 편의 함수들
def create_yolov5(size: str = 'n', pretrained: bool = True) -> YOLO:
    """YOLOv5 모델 생성"""
    return ModelFactory.create_model('yolov5', size, pretrained)


def create_yolov8(size: str = 'n', pretrained: bool = True) -> YOLO:
    """YOLOv8 모델 생성"""
    return ModelFactory.create_model('yolov8', size, pretrained)


def create_yolov11(size: str = 'n', pretrained: bool = True) -> YOLO:
    """YOLOv11 모델 생성"""
    return ModelFactory.create_model('yolov11', size, pretrained)


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)

    # 지원 모델 확인
    logger.info("=" * 50)
    logger.info("Supported models:")
    logger.info(ModelFactory.get_supported_models())
    logger.info("=" * 50)

    # 동적으로 여러 모델 생성 테스트
    models_to_test = [
        ('yolov8', 'n'),
        ('yolov5', 's'),
        ('yolov8', 'm'),
    ]

    for model_type, model_size in models_to_test:
        logger.info(f"--- Testing {model_type.upper()}{model_size.upper()} ---")
        try:
            # ModelFactory를 직접 사용
            model = ModelFactory.create_model(model_type, model_size)
            logger.info(f"Successfully created via ModelFactory.")
            logger.info("Model info:")
            logger.info(get_model_info(model))

            # 편의 함수 사용 (해당하는 경우)
            if model_type == 'yolov8':
                model_from_func = create_yolov8(model_size)
                logger.info(f"Successfully created via create_yolov8('{model_size}').")
            elif model_type == 'yolov5':
                model_from_func = create_yolov5(model_size)
                logger.info(f"Successfully created via create_yolov5('{model_size}').")

        except Exception as e:
            logger.error(f"Failed to create {model_type}{model_size}: {e}")
        logger.info("-" * 50)