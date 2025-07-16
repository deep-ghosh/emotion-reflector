import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Create a simple model architecture (this will be untrained but demonstrates the structure)
def create_model():
    model = Sequential()
    # convolutional layers
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    # fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# For demonstration, let's create a basic emotion detector using face detection
def detect_basic_emotion(face_roi):
    """
    Basic emotion detection based on simple image features
    This is a placeholder - in reality you'd use a trained model
    """
    # Simple heuristic based on image statistics (not accurate, just for demo)
    mean_intensity = np.mean(face_roi)
    std_intensity = np.std(face_roi)
    
    # Very basic classification (not accurate, just for demonstration)
    if mean_intensity < 80:
        return "sad"
    elif mean_intensity > 180:
        return "happy"
    elif std_intensity > 50:
        return "surprise"
    else:
        return "neutral"

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Initialize camera and face detection
webcam = cv2.VideoCapture(0)
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Create model (untrained for now)
model = create_model()
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

print("Starting basic emotion detection...")
print("Press 'q' to quit")
print("Note: This is a demo version. For accurate results, you need a trained model.")

while True:
    ret, frame = webcam.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Use basic emotion detection (placeholder)
            emotion = detect_basic_emotion(face_roi)
            
            # Display emotion label
            cv2.putText(frame, emotion, (x-10, y-10), 
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
            # Also show that we're ready for proper model
            cv2.putText(frame, "Demo Mode - Need Trained Model", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Face Emotion Detection (Demo)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
print("Demo completed!")
