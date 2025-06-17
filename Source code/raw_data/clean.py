import os
import shutil
from pathlib import Path

# Define paths
source_image_folder = Path("VBS/raw_data/data/dataset/images")  # the folder with all images (train/val merged if needed)
source_label_folder = Path("VBS/raw_data/data/dataset/raw_data_labels")  # all your label files

target_image_folder = Path("VBS/clean_dataset/images")
target_label_folder = Path("VBS/clean_dataset/labels")

# Create target folders
target_image_folder.mkdir(parents=True, exist_ok=True)
target_label_folder.mkdir(parents=True, exist_ok=True)

# Supported image formats
image_extensions = [".jpg", ".jpeg", ".png"]

# Process all image files
for image_path in source_image_folder.rglob("*"):
    if image_path.suffix.lower() in image_extensions:
        label_filename = image_path.stem + ".txt"
        label_path = source_label_folder / label_filename

        if label_path.exists():
            # Copy image and label
            shutil.copy2(image_path, target_image_folder / image_path.name)
            shutil.copy2(label_path, target_label_folder / label_filename)
            print(f"✅ Copied: {image_path.name} & {label_filename}")
        else:
            print(f"⚠️  Skipped (no label): {image_path.name}")

print("\n✅ Finished organizing dataset into 'clean_dataset/images' and 'clean_dataset/labels'")
