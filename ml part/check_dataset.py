
from pathlib import Path

dataset_path = Path(
    r"C:\Users\ganesh\Desktop\emotion\datasetemotion"
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "test"]:
    folder_path = dataset_path / split
    print(f"\n--- {split.upper()} DATASET ---")

    if not folder_path.exists():
        print("Folder not found:", folder_path)
        continue

    total = 0

    for emotion_folder in sorted(folder_path.iterdir()):
        if emotion_folder.is_dir():
            count = sum(
                1 for file in emotion_folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in extensions
            )
            total += count
            print(f"{emotion_folder.name}: {count} images")

    print("TOTAL:", total)
