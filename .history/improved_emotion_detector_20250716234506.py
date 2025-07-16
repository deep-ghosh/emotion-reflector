"""
Improved Face Emotion Detection with Better Feature Analysis
This version uses multiple face analysis techniques for better emotion recognition
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

class ImprovedEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize MediaPipe Face Mesh for better face analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load model if available
        self.model = self.load_model()
        
        # Enhanced emotion detection using facial landmarks
        self.emotion_weights = {
            'angry': {'eyebrows': 0.4, 'eyes': 0.3, 'mouth': 0.3},
            'sad': {'eyebrows': 0.2, 'eyes': 0.4, 'mouth': 0.4},
            'happy': {'eyebrows': 0.1, 'eyes': 0.3, 'mouth': 0.6},
            'surprise': {'eyebrows': 0.5, 'eyes': 0.3, 'mouth': 0.2},
            'fear': {'eyebrows': 0.4, 'eyes': 0.4, 'mouth': 0.2},
            'disgust': {'eyebrows': 0.3, 'eyes': 0.2, 'mouth': 0.5},
            'neutral': {'eyebrows': 0.2, 'eyes': 0.3, 'mouth': 0.5}
        }
    
    def load_model(self):
        """Load the emotion detection model if available"""
        try:
            if os.path.exists('facialemotionmodel.json'):
                with open('facialemotionmodel.json', 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json)
                
                # Try to load weights
                if os.path.exists('facialemotionmodel.weights.h5'):
                    model.load_weights('facialemotionmodel.weights.h5')
                    print("✅ Model loaded successfully")
                    return model
                else:
                    print("⚠️  Model architecture loaded but no trained weights found")
                    return None
            else:
                print("⚠️  No model file found")
                return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def analyze_facial_features(self, image, landmarks):
        """Analyze facial features using MediaPipe landmarks"""
        if not landmarks:
            return None
            
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Extract key facial points
        face_landmarks = landmarks.landmark
        
        # Eyebrow analysis (landmarks around eyebrows)
        left_eyebrow = [face_landmarks[70], face_landmarks[63], face_landmarks[105]]
        right_eyebrow = [face_landmarks[300], face_landmarks[293], face_landmarks[334]]
        
        # Eye analysis (landmarks around eyes)
        left_eye = [face_landmarks[33], face_landmarks[7], face_landmarks[163], face_landmarks[144]]
        right_eye = [face_landmarks[362], face_landmarks[382], face_landmarks[381], face_landmarks[373]]
        
        # Mouth analysis (landmarks around mouth)
        mouth = [face_landmarks[78], face_landmarks[81], face_landmarks[13], face_landmarks[311], face_landmarks[308]]
        
        # Calculate feature ratios
        features = self.calculate_feature_ratios(left_eyebrow, right_eyebrow, left_eye, right_eye, mouth, h, w)
        
        return features
    
    def calculate_feature_ratios(self, left_eyebrow, right_eyebrow, left_eye, right_eye, mouth, h, w):
        """Calculate ratios for emotion detection"""
        features = {}
        
        # Eyebrow height ratio (higher = surprise/fear, lower = angry/sad)
        left_eyebrow_y = sum([point.y for point in left_eyebrow]) / len(left_eyebrow)
        right_eyebrow_y = sum([point.y for point in right_eyebrow]) / len(right_eyebrow)
        eyebrow_height = (left_eyebrow_y + right_eyebrow_y) / 2
        features['eyebrow_height'] = eyebrow_height
        
        # Eye openness ratio (wider = surprise, narrower = angry/disgust)
        left_eye_height = abs(left_eye[1].y - left_eye[3].y)
        right_eye_height = abs(right_eye[1].y - right_eye[3].y)
        eye_openness = (left_eye_height + right_eye_height) / 2
        features['eye_openness'] = eye_openness
        
        # Mouth curve ratio (higher = happy, lower = sad)
        mouth_corners = [mouth[0].y, mouth[4].y]  # Left and right corners
        mouth_center = mouth[2].y  # Center of mouth
        mouth_curve = (sum(mouth_corners) / 2) - mouth_center
        features['mouth_curve'] = mouth_curve
        
        # Mouth width ratio
        mouth_width = abs(mouth[0].x - mouth[4].x)
        features['mouth_width'] = mouth_width
        
        return features
    
    def predict_emotion_from_features(self, features):
        """Predict emotion based on facial features"""
        if not features:
            return 'neutral', 0.5
            
        emotion_scores = {}
        
        # Analyze each emotion based on feature patterns
        for emotion in self.emotion_labels:
            score = 0
            
            if emotion == 'angry':
                # Angry: low eyebrows, narrow eyes, tight mouth
                score += (1 - features['eyebrow_height']) * 0.4  # Lower eyebrows
                score += (1 - features['eye_openness']) * 0.3    # Narrower eyes
                score += (1 - features['mouth_width']) * 0.3     # Tighter mouth
                
            elif emotion == 'sad':
                # Sad: drooping features, downturned mouth
                score += features['eyebrow_height'] * 0.2        # Slightly raised inner eyebrows
                score += (1 - features['eye_openness']) * 0.4    # Droopy eyes
                score += (1 - features['mouth_curve']) * 0.4     # Downturned mouth
                
            elif emotion == 'happy':
                # Happy: raised mouth corners, crinkled eyes
                score += features['mouth_curve'] * 0.6           # Upturned mouth
                score += (1 - features['eye_openness']) * 0.3    # Squinted eyes from smiling
                score += 0.1  # Base happiness score
                
            elif emotion == 'surprise':
                # Surprise: raised eyebrows, wide eyes, open mouth
                score += (1 - features['eyebrow_height']) * 0.5  # Raised eyebrows
                score += features['eye_openness'] * 0.3          # Wide eyes
                score += features['mouth_width'] * 0.2           # Open mouth
                
            elif emotion == 'fear':
                # Fear: raised eyebrows, wide eyes, tense mouth
                score += (1 - features['eyebrow_height']) * 0.4  # Raised eyebrows
                score += features['eye_openness'] * 0.4          # Wide eyes
                score += (1 - features['mouth_curve']) * 0.2     # Tense mouth
                
            elif emotion == 'disgust':
                # Disgust: wrinkled nose area, raised upper lip
                score += (1 - features['eyebrow_height']) * 0.3  # Slightly lowered eyebrows
                score += (1 - features['eye_openness']) * 0.2    # Squinted eyes
                score += features['mouth_curve'] * 0.5           # Raised upper lip
                
            else:  # neutral
                # Neutral: balanced features
                score += (1 - abs(features['eyebrow_height'] - 0.5)) * 0.3
                score += (1 - abs(features['eye_openness'] - 0.5)) * 0.3
                score += (1 - abs(features['mouth_curve'])) * 0.4
            
            emotion_scores[emotion] = max(0, min(1, score))
        
        # Get the emotion with highest score
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[best_emotion]
        
        return best_emotion, confidence
    
    def extract_features(self, image):
        """Extract features from face image"""
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Analyze facial features
                features = self.analyze_facial_features(image, results.multi_face_landmarks[0])
                emotion, confidence = self.predict_emotion_from_features(features)
                return emotion, confidence
            else:
                return 'neutral', 0.5
                
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return 'neutral', 0.5
    
    def detect_emotion(self, image):
        """Main emotion detection function"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Predict emotion using feature analysis
            emotion, confidence = self.extract_features(face_roi)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence
            })
        
        return results
    
    def run_realtime_detection(self):
        """Run real-time emotion detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎭 Improved Emotion Detection Started!")
        print("📷 Camera is running - press 'q' to quit")
        print("🎯 This version uses facial feature analysis for better emotion recognition")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect emotions
            results = self.detect_emotion(frame)
            
            # Draw results
            for result in results:
                x, y, w, h = result['bbox']
                emotion = result['emotion']
                confidence = result['confidence']
                
                # Draw rectangle around face
                color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{emotion.upper()}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Add emotion-specific visual feedback
                if emotion == 'angry':
                    cv2.putText(frame, "😠", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif emotion == 'sad':
                    cv2.putText(frame, "😢", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif emotion == 'happy':
                    cv2.putText(frame, "😊", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif emotion == 'surprise':
                    cv2.putText(frame, "😮", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif emotion == 'fear':
                    cv2.putText(frame, "😨", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                elif emotion == 'disgust':
                    cv2.putText(frame, "🤢", (x+w-50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 128), 2)
            
            # Display frame
            cv2.imshow('Improved Emotion Detection', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎯 Improved emotion detection stopped")

def main():
    detector = ImprovedEmotionDetector()
    detector.run_realtime_detection()

if __name__ == "__main__":
    main()
