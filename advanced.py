import face_recognition
import cv2
import os
import pyttsx3
import numpy as np
import pytesseract
from PIL import Image
import threading

# Configuration
KNOWN_FACES_DIR = "KNOWN_FACES"
KNOWN_OBJECTS_DIR = "KNOWN_OBJECTS"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(KNOWN_OBJECTS_DIR, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Text detection variables
last_spoken_text = ""
text_cooldown = False

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Object detection setup (YOLOv4)
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Mode selection
MODES = {
    'face': "Face Recognition Mode",
    'text': "Text Recognition Mode",
    'object': "Object Detection Mode"
}
current_mode = 'face'

def speak(text, interrupt=False):
    if interrupt:
        engine.stop()
    engine.say(text)
    engine.runAndWait()

def reset_text_cooldown():
    global text_cooldown
    text_cooldown = False

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

def load_known_objects():
    known_objects = {}
    for filename in os.listdir(KNOWN_OBJECTS_DIR):
        if filename.lower().endswith(('jpg', 'png', 'jpeg')):
            name = os.path.splitext(filename)[0]
            known_objects[name] = cv2.imread(os.path.join(KNOWN_OBJECTS_DIR, filename))
    return known_objects

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id != 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detected_objects

# Load initial databases
known_face_encodings, known_face_names = load_known_faces()
known_objects = load_known_objects()
spoken_names = set()
detected_objects_cache = set()

# Start video capture
video_capture = cv2.VideoCapture(0)
speak("Application started. Current mode: " + MODES[current_mode])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Only flip for face mode (remove mirror effect for text/object modes)
    display_frame = frame.copy()
    if current_mode == 'face':
        display_frame = cv2.flip(frame, 1)
    
    cv2.putText(display_frame, MODES[current_mode], (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if current_mode == 'face':
        small_frame = cv2.resize(display_frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        current_names = []
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            current_names.append(name)
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            if name != "Unknown" and name not in spoken_names:
                speak(name)  # Just says the name without prefix
                spoken_names.add(name)
        
        if "Unknown" in current_names and not spoken_names:
            cv2.imshow("New Face Detected", display_frame)
            speak("Please name this person")
            new_name = input("Enter name: ").strip()
            if new_name:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.jpg")
                cv2.imwrite(save_path, frame)
                speak(f"{new_name} saved")
                known_face_encodings, known_face_names = load_known_faces()
                spoken_names.clear()
    
    elif current_mode == 'text':
        # Use original unflipped frame for text recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Improved Tesseract configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config).strip()
        
        if text and text != last_spoken_text and not text_cooldown:
            speak(text)  # Just says the text without "Text detected" prefix
            last_spoken_text = text
            text_cooldown = True
            cv2.putText(display_frame, "Text Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            threading.Timer(3.0, reset_text_cooldown).start()
        elif not text:
            last_spoken_text = ""
    
    elif current_mode == 'object':
        processed_frame, detected_objects = detect_objects(frame.copy())
        display_frame = processed_frame
        
        new_objects = [obj for obj in detected_objects if obj not in detected_objects_cache]
        if new_objects:
            speak(", ".join(new_objects))  # Just says object names without prefix
            detected_objects_cache.update(new_objects)
    
    cv2.imshow('Accessibility Assistant', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        current_mode = 'face'
        spoken_names.clear()
        speak(MODES[current_mode])
    elif key == ord('t'):
        current_mode = 'text'
        last_spoken_text = ""
        speak(MODES[current_mode])
    elif key == ord('o'):
        current_mode = 'object'
        detected_objects_cache.clear()
        speak(MODES[current_mode])
    elif key == ord('s') and current_mode == 'object':
        speak("Please name the object")
        obj_name = input("Enter object name: ").strip()
        if obj_name:
            save_path = os.path.join(KNOWN_OBJECTS_DIR, f"{obj_name}.jpg")
            cv2.imwrite(save_path, frame)
            speak(f"{obj_name} saved")
            known_objects = load_known_objects()

video_capture.release()
cv2.destroyAllWindows()
speak("Application closed")