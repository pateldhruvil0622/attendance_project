# clean_dataset.py

from PIL import Image
import numpy as np
import os

DATASET_DIR = "dataset"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg')

print("🧹 Starting dataset cleanup...")

for student in os.listdir(DATASET_DIR):
    student_folder = os.path.join(DATASET_DIR, student)
    if not os.path.isdir(student_folder):
        continue

    print(f"\n🎓 Processing student: {student}")

    for img_file in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_file)

        if not img_file.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"🗑️ Removed: {img_file}")
            os.remove(img_path)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            img_array = np.array(image)

            if img_array.dtype != "uint8":
                img_array = (img_array / 256).astype("uint8")

            fixed_image = Image.fromarray(img_array)
            fixed_image.thumbnail((800, 800))
            fixed_image.save(img_path, "JPEG", quality=90)
            print(f"✅ Cleaned: {img_file}")

        except Exception as e:
            print(f"❌ Failed to process {img_file}: {e}")
            os.remove(img_path)

print("\n✅ Dataset cleanup complete.")