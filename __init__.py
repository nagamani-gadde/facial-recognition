import os
import django
import sys

# Set up Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_project.settings')
django.setup()

# Import your model
from face_app.models import CaptureFace
import cv2
import os
from datetime import datetime

# Create media folder if it doesn't exist
if not os.path.exists('media/faces'):
    os.makedirs('media/faces')

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open your webcam (0 is default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image
        face_img = frame[y:y+h, x:x+w]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"media/faces/face_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"Face saved to {filename}")

    # Show the video feed
    cv2.imshow("Face Detector - Press 'q' to Quit", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from django.db import models

class CapturedFace(models.Model):
    image = models.ImageField(upload_to='faces/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Captured at {self.timestamp}"
from django.contrib import admin
from .models import CapturedFace

admin.site.register(CapturedFace)
import cv2
import os
import django
import uuid

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_project.settings")
django.setup()

from face_app.models import CapturedFace
from django.core.files import File

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Starting camera. Press 'c' to capture face or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Capture', frame)

    key = cv2.waitKey(1)
    if key == ord('c'):  # Capture
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            filename = f"face_{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("media/faces", filename)

            # Ensure folder exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save face image
            cv2.imwrite(filepath, face_img)

            # Save in Django model
            with open(filepath, 'rb') as f:
                captured = CapturedFace()
                captured.image.save(filename, File(f), save=True)

            print(f"Face saved: {filename}")

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()


