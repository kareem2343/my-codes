import face_recognition
import cv2
import os
import pyttsx3
import numpy as np

KNOWN_FACES_DIR = "KNOWN_FACES"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('jpg', 'png', 'jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()
spoken_names = set()

video_capture = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame to 1/4 size to save memory
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name != "Unknown":
            if name not in spoken_names:
                spoken_names.add(name)
                speak(f"Hello {name}")
        else:
            cv2.imshow("New Face Detected", frame)
            speak("Unknown person detected. Please enter a name.")
            new_name = input("Enter name: ").strip()
            if new_name:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")
                cv2.imwrite(save_path, frame)
                speak(f"Saved as {new_name}")
                known_face_encodings, known_face_names = load_known_faces()

        # Scale back for display
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()