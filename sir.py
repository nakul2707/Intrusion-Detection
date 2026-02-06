import cv2
import os

# Load OpenCV's pre-trained Haar face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video file path
video_path = "input7.mp4"
cap = cv2.VideoCapture(video_path)

# Output folder
output_folder = "captured_faces"
os.makedirs(output_folder, exist_ok=True)

face_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    h0, w0 = frame.shape[:2]
    new_w   = 640
    new_h   = int(h0 * new_w / w0)        # scale height to keep same aspect ratio
    resized = cv2.resize(frame, (new_w, new_h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = resized[y:y + h, x:x + w]
        face_file = os.path.join(output_folder, f"face_{face_id:03}.jpg")
        cv2.imwrite(face_file, face_crop)
        face_id += 1

        # Draw box
        cv2.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show result
    cv2.imshow('Face Capture', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Extracted {face_id} faces into '{output_folder}'")