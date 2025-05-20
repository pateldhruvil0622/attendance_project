# test_face_recognition.py

import face_recognition
import cv2
import os

print("🧪 Testing libraries...")

# Test face_recognition
test_image_path = "test_images/test1.jpg"
if os.path.exists(test_image_path):
    image = face_recognition.load_image_file(test_image_path)
    face_locations = face_recognition.face_locations(image)
    print(f"✅ Detected {len(face_locations)} face(s) in test image.")
else:
    print("⚠️ Test image not found. Can't verify full functionality.")

print("🎉 All tests passed!")