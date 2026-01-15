import logging
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Quantizer:
    def __init__(self, weights_path: str, output_dir: Optional[str] = None):
        """
        Args:
            weights_path: 파인튜닝이 완료된 .pt 모델 경로
            output_dir: quantized 모델 저장 디렉토리
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        self.output_dir = Path(output_dir) if output_dir else self.weights_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.weights_path))
        logger.info(f"Loaded trained model from {self.weights_path}")

    def export_fp16(self):
        """
        FP16 Quantization (가장 안전, 성능 손실 거의 없음)
        """
        logger.info("Starting FP16 quantization export...")
        self.model.export(
            format="engine",
            half=True, # FP32 → FP16
            device=0
        )
        logger.info("FP16 quantization export completed")

    def export_int8(self, data_yaml: str, imgsz: int = 640):
        """
        INT8 Quantization (Calibration 필요)

        Args:
            data_yaml: calibration용 dataset yaml 경로
            imgsz: 입력 이미지 크기
        """
        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

        logger.info("Starting INT8 quantization export...")
        self.model.export(
            format="engine",
            int8=True,
            data=str(data_yaml),
            imgsz=imgsz,
            device=0
        )
        logger.info("INT8 quantization export completed")


def quantize_model(
    weights_path: str,
    mode: str = "fp16",
    data_yaml: Optional[str] = None,
    imgsz: int = 640
):
    """
    편의 함수

    Args:
        weights_path: 파인튜닝 완료된 모델 경로
        mode: 'fp16' or 'int8'
        data_yaml: INT8일 경우 calibration dataset yaml
        imgsz: 입력 이미지 크기
    """
    quantizer = Quantizer(weights_path)

    if mode == "fp16":
        quantizer.export_fp16()
    elif mode == "int8":
        if data_yaml is None:
            raise ValueError("INT8 quantization requires data_yaml for calibration")
        quantizer.export_int8(data_yaml=data_yaml, imgsz=imgsz)
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="YOLO Post-Training Quantization")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--mode", type=str, choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--data", type=str, help="Dataset yaml (required for INT8)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    quantize_model(
        weights_path=args.weights,
        mode=args.mode,
        data_yaml=args.data,
        imgsz=args.imgsz
    )