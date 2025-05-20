# check_and_convert_images.py

from PIL import Image
import numpy as np
import os

DATASET_DIR = "dataset"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg')

print("🔍 Checking and converting images...\n")

for student in os.listdir(DATASET_DIR):
    student_folder = os.path.join(DATASET_DIR, student)
    if not os.path.isdir(student_folder):
        continue

    print(f"🎓 Processing student: {student}")

    for img_file in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_file)

        if not img_file.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"🗑️ Removed non-JPG file: {img_file}")
            os.remove(img_path)
            continue

        try:
            # Open image
            image = Image.open(img_path)

            # Confirm it can be read
            image.verify()
            image = Image.open(img_path)

            # Convert to RGB mode
            image = image.convert("RGB")

            # Convert to NumPy array to check bit depth
            img_array = np.array(image)

            if img_array.dtype != "uint8":
                print(f"⚠️ Converting {img_file} from {img_array.dtype} to uint8")
                img_array = (img_array / 256).astype("uint8")

            # Resize for performance
            image = Image.fromarray(img_array)
            image.thumbnail((800, 800))

            # Save back as clean JPEG
            image.save(img_path, "JPEG", quality=90)
            print(f"✅ Cleaned: {img_file}")

        except Exception as e:
            print(f"❌ Removing invalid image: {img_file} → {e}")
            os.remove(img_path)

print("\n✅ All images checked and converted.")