# face_recognition_attendance.py

import face_recognition
import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

# Paths
ENCODINGS_FILE = "known_faces/encodings.pkl"
TEST_IMAGE_PATH = "test_images/test1.jpg"
ATTENDANCE_CSV = "attendance.csv"

# Load known faces
if not os.path.exists(ENCODINGS_FILE):
    print("❌ Encodings file not found. Please run train_faces.py first.")
    exit()

with open(ENCODINGS_FILE, "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Load test image
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
    exit()

image = cv2.imread(TEST_IMAGE_PATH)
if image is None:
    print("❌ Failed to load image. Check file format.")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Initialize attendance DataFrame
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
else:
    df = pd.DataFrame(columns=["Name", "Timestamp"])

# Loop through each face found
for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    name = "Unknown"

    if len(face_distances) > 0:
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_names[best_match_index]

            if name not in df.values:
                new_entry = {"Name": name, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                df = df.append(new_entry, ignore_index=True)
                print(f"✅ {name} marked present.")

    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save attendance
df.to_csv(ATTENDANCE_CSV, index=False)
print(f"📊 Attendance saved to '{ATTENDANCE_CSV}'")

# Show result
cv2.imshow("Face Recognition Result", image)
print("🖼️ Displaying result. Press any key to close window.")
cv2.waitKey(0)
cv2.destroyAllWindows()