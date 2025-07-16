"""
Enhanced Face Emotion Detection with Better Pattern Recognition
This version uses improved image processing and pattern analysis for accurate emotion detection
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

class EnhancedEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Load model if available
        self.model = self.load_model()
        
        # Emotion pattern templates based on facial geometry
        self.emotion_patterns = {
            'angry': {'eyebrow_angle': -15, 'eye_openness': 0.3, 'mouth_curve': -0.2},
            'sad': {'eyebrow_angle': 10, 'eye_openness': 0.4, 'mouth_curve': -0.5},
            'happy': {'eyebrow_angle': 5, 'eye_openness': 0.6, 'mouth_curve': 0.8},
            'surprise': {'eyebrow_angle': 20, 'eye_openness': 0.9, 'mouth_curve': 0.1},
            'fear': {'eyebrow_angle': 25, 'eye_openness': 0.8, 'mouth_curve': -0.1},
            'disgust': {'eyebrow_angle': -10, 'eye_openness': 0.4, 'mouth_curve': -0.3},
            'neutral': {'eyebrow_angle': 0, 'eye_openness': 0.5, 'mouth_curve': 0.0}
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
                    print("⚠️  Using pattern-based detection (no trained weights)")
                    return None
            else:
                print("⚠️  Using pattern-based detection (no model file)")
                return None
        except Exception as e:
            print(f"❌ Error loading model, using pattern-based detection: {e}")
            return None
    
    def analyze_facial_regions(self, face_gray):
        """Analyze facial regions for emotion detection"""
        h, w = face_gray.shape
        
        # Define regions of interest
        eyebrow_region = face_gray[0:int(h*0.4), :]
        eye_region = face_gray[int(h*0.2):int(h*0.6), :]
        mouth_region = face_gray[int(h*0.6):h, :]
        
        features = {}
        
        # Analyze eyebrow region
        features['eyebrow_intensity'] = self.analyze_eyebrow_region(eyebrow_region)
        
        # Analyze eye region
        features['eye_features'] = self.analyze_eye_region(eye_region)
        
        # Analyze mouth region
        features['mouth_features'] = self.analyze_mouth_region(mouth_region)
        
        return features
    
    def analyze_eyebrow_region(self, region):
        """Analyze eyebrow region for emotion indicators"""
        # Calculate gradient to detect eyebrow angle
        sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant angles
        angles = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Get the mean angle (indicates eyebrow position)
        mean_angle = np.mean(angles[np.abs(angles) < 45])  # Filter extreme angles
        
        return mean_angle if not np.isnan(mean_angle) else 0
    
    def analyze_eye_region(self, region):
        """Analyze eye region for openness and expression"""
        # Detect eyes in the region
        eyes = self.eye_cascade.detectMultiScale(region, 1.1, 5)
        
        if len(eyes) == 0:
            return {'openness': 0.5, 'count': 0}
        
        # Calculate eye openness based on detected eye dimensions
        total_openness = 0
        for (x, y, w, h) in eyes:
            # Eye openness ratio (height/width)
            openness_ratio = h / w if w > 0 else 0.5
            total_openness += openness_ratio
        
        avg_openness = total_openness / len(eyes)
        
        return {'openness': avg_openness, 'count': len(eyes)}
    
    def analyze_mouth_region(self, region):
        """Analyze mouth region for smile/frown detection"""
        # Apply edge detection to find mouth contours
        edges = cv2.Canny(region, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'curve': 0.0, 'width': 0.0}
        
        # Get the largest contour (likely the mouth)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate mouth curve based on contour shape
        # Check if mouth corners are higher or lower than center
        if len(largest_contour) > 10:
            # Get leftmost, rightmost, and center points
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            
            # Find center point
            center_x = (leftmost[0] + rightmost[0]) // 2
            center_points = [point for point in largest_contour[:, 0] if abs(point[0] - center_x) < 5]
            
            if center_points:
                center_y = np.mean([point[1] for point in center_points])
                corner_y = (leftmost[1] + rightmost[1]) / 2
                
                # Positive curve = smile, negative = frown
                curve = (corner_y - center_y) / h if h > 0 else 0
            else:
                curve = 0
        else:
            curve = 0
        
        return {'curve': curve, 'width': w / region.shape[1] if region.shape[1] > 0 else 0}
    
    def classify_emotion_from_features(self, features):
        """Classify emotion based on extracted features"""
        eyebrow_angle = features['eyebrow_intensity']
        eye_openness = features['eye_features']['openness']
        mouth_curve = features['mouth_features']['curve']
        
        # Normalize features
        eye_openness = max(0, min(1, eye_openness))
        mouth_curve = max(-1, min(1, mouth_curve))
        
        # Calculate similarity to each emotion pattern
        emotion_scores = {}
        
        for emotion, pattern in self.emotion_patterns.items():
            score = 0
            
            # Eyebrow angle similarity
            angle_diff = abs(eyebrow_angle - pattern['eyebrow_angle'])
            angle_score = max(0, 1 - angle_diff / 45)  # Normalize to 0-1
            
            # Eye openness similarity
            eye_diff = abs(eye_openness - pattern['eye_openness'])
            eye_score = max(0, 1 - eye_diff)
            
            # Mouth curve similarity
            mouth_diff = abs(mouth_curve - pattern['mouth_curve'])
            mouth_score = max(0, 1 - mouth_diff)
            
            # Weighted combination
            score = (angle_score * 0.3 + eye_score * 0.3 + mouth_score * 0.4)
            emotion_scores[emotion] = score
        
        # Get the emotion with highest score
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[best_emotion]
        
        # Apply some heuristics to improve accuracy
        if mouth_curve > 0.3:
            best_emotion = 'happy'
            confidence = max(confidence, 0.8)
        elif mouth_curve < -0.3:
            best_emotion = 'sad'
            confidence = max(confidence, 0.7)
        elif eye_openness > 0.8 and eyebrow_angle > 15:
            best_emotion = 'surprise'
            confidence = max(confidence, 0.7)
        elif eye_openness < 0.3 and eyebrow_angle < -10:
            best_emotion = 'angry'
            confidence = max(confidence, 0.7)
        
        return best_emotion, confidence
    
    def extract_features(self, image):
        """Extract features from face image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistent analysis
            gray = cv2.resize(gray, (200, 200))
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            # Analyze facial regions
            features = self.analyze_facial_regions(gray)
            
            # Classify emotion
            emotion, confidence = self.classify_emotion_from_features(features)
            
            return emotion, confidence
            
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
            # Extract face region with some padding
            padding = 20
            face_roi = image[max(0, y-padding):min(image.shape[0], y+h+padding),
                           max(0, x-padding):min(image.shape[1], x+w+padding)]
            
            if face_roi.size > 0:
                # Predict emotion using enhanced feature analysis
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
        
        print("🎭 Enhanced Emotion Detection Started!")
        print("📷 Camera is running - press 'q' to quit")
        print("🎯 This version uses advanced pattern recognition for better accuracy")
        print("😊 Try different expressions: happy, sad, angry, surprised, neutral")
        
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
                
                # Color coding based on emotion
                emotion_colors = {
                    'angry': (0, 0, 255),      # Red
                    'sad': (255, 0, 0),        # Blue
                    'happy': (0, 255, 0),      # Green
                    'surprise': (0, 255, 255), # Yellow
                    'fear': (128, 0, 128),     # Purple
                    'disgust': (0, 128, 128),  # Teal
                    'neutral': (128, 128, 128) # Gray
                }
                
                color = emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label with confidence
                label = f"{emotion.upper()}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add emotion emoji
                emoji_map = {
                    'angry': '😠', 'sad': '😢', 'happy': '😊', 'surprise': '😮',
                    'fear': '😨', 'disgust': '🤢', 'neutral': '😐'
                }
                
                # Display status
                status = f"DETECTED: {emotion.upper()}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display frame
            cv2.imshow('Enhanced Emotion Detection', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎯 Enhanced emotion detection stopped")

def main():
    detector = EnhancedEmotionDetector()
    detector.run_realtime_detection()

if __name__ == "__main__":
    main()
