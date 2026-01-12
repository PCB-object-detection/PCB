import kagglehub
import shutil
from pathlib import Path

# Download latest version
print("Downloading PCB Defect Dataset from Kaggle...")
cache_path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
print(f"✅ Downloaded to cache: {cache_path}")

# dataset/raw/ 디렉토리 경로
raw_dir = Path(__file__).parent.parent / "dataset" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# 캐시에서 dataset/raw/로 복사
print(f"\nCopying to {raw_dir}...")

# kagglehub는 pcb-defect-dataset 폴더 안에 실제 데이터를 넣음
# 그 안의 내용(train, test, val, data.yaml)을 dataset/raw/ 안으로 복사
cache_path = Path(cache_path)
source_files = list(cache_path.iterdir())

for item in source_files:
    dest = raw_dir / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)
    print(f"  ✓ Copied: {item.name}")

print("\n✅ Dataset setup complete!")

# 최종 구조 출력
print(f"\nFiles in dataset/raw/:")
for item in sorted(raw_dir.iterdir()):
    if item.name != '.gitkeep':
        print(f"  - {item.name}")


