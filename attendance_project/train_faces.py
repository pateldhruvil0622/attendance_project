# train_faces.py

import os
import face_recognition
from PIL import Image
import numpy as np
import pickle

print("Starting training process...")

known_encodings = []
known_names = []

DATASET_DIR = "dataset"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg')

for student_name in os.listdir(DATASET_DIR):
    student_folder = os.path.join(DATASET_DIR, student_name)

    if not os.path.isdir(student_folder):
        continue

    print(f"\n🎓 Training on student: {student_name}")

    for image_file in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_file)

        if not image_file.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"🚫 Skipping non-JPG file: {image_file}")
            continue

        try:
            # Open image with PIL
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)

            if img_array.dtype != "uint8":
                print(f"⚠️ Converting {image_file} from {img_array.dtype} to uint8")
                img_array = (img_array / 256).astype("uint8")

            fixed_image = Image.fromarray(img_array)
            fixed_image.thumbnail((800, 800))
            fixed_image.save(image_path, "JPEG", quality=90)

            # Load image for face_recognition
            image_data = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image_data)

            if face_encodings:
                known_encodings.append(face_encodings[0])
                known_names.append(student_name)
                print(f"📸 Encoded face from: {image_file}")
            else:
                print(f"❌ No face found in: {image_file}")

        except Exception as e:
            print(f"⚠️ Error processing {image_file}: {str(e)}")
            continue

# Save encodings
OUTPUT_PATH = "known_faces/encodings.pkl"

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("\n✅ Training complete.")
print(f"💾 Encodings saved to: {OUTPUT_PATH}")