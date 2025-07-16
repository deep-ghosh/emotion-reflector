"""
Advanced Face Emotion Detection with Pattern Recognition
This version uses computer vision techniques without TensorFlow dependency
"""

import cv2
import numpy as np

class AdvancedEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        print("🎭 Advanced Emotion Detection System Initialized")
        print("⚠️  Using pattern-based detection for accurate emotion recognition")
    
    def analyze_eyebrows(self, face_gray):
        """Analyze eyebrow region for emotion indicators"""
        h, w = face_gray.shape
        eyebrow_region = face_gray[0:int(h*0.35), :]
        
        # Apply edge detection to find eyebrow lines
        edges = cv2.Canny(eyebrow_region, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'angle': 0, 'height': 0.5}
        
        # Calculate average angle of eyebrow contours
        angles = []
        for contour in contours:
            if len(contour) > 10:  # Filter small contours
                # Fit line to contour
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.arctan2(vy, vx) * 180 / np.pi
                angles.append(angle)
        
        avg_angle = np.mean(angles) if angles else 0
        
        # Calculate eyebrow height (position relative to face)
        eyebrow_height = np.mean(np.nonzero(edges)[0]) / h if np.any(edges) else 0.5
        
        return {'angle': avg_angle, 'height': eyebrow_height}
    
    def analyze_eyes(self, face_gray):
        """Analyze eye region for openness and characteristics"""
        h, w = face_gray.shape
        eye_region = face_gray[int(h*0.2):int(h*0.6), :]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 5)
        
        if len(eyes) == 0:
            return {'openness': 0.5, 'count': 0, 'symmetry': 0.5}
        
        # Calculate eye characteristics
        eye_openness = []
        eye_positions = []
        
        for (x, y, w_eye, h_eye) in eyes:
            # Eye openness ratio
            openness = h_eye / w_eye if w_eye > 0 else 0.5
            eye_openness.append(openness)
            eye_positions.append((x + w_eye//2, y + h_eye//2))
        
        avg_openness = np.mean(eye_openness)
        
        # Eye symmetry (how aligned the eyes are)
        symmetry = 0.5
        if len(eye_positions) >= 2:
            # Calculate vertical alignment
            y_diff = abs(eye_positions[0][1] - eye_positions[1][1])
            symmetry = max(0, 1 - y_diff / 20)  # Normalize
        
        return {
            'openness': avg_openness,
            'count': len(eyes),
            'symmetry': symmetry
        }
    
    def analyze_mouth(self, face_gray):
        """Analyze mouth region for emotion indicators"""
        h, w = face_gray.shape
        mouth_region = face_gray[int(h*0.55):h, :]
        
        # Apply adaptive thresholding for better mouth detection
        mouth_thresh = cv2.adaptiveThreshold(mouth_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'curve': 0, 'openness': 0.5, 'width': 0.5}
        
        # Get the largest contour (likely the mouth)
        mouth_contour = max(contours, key=cv2.contourArea)
        
        # Calculate mouth characteristics
        x, y, w_mouth, h_mouth = cv2.boundingRect(mouth_contour)
        
        # Mouth curve analysis
        if len(mouth_contour) > 10:
            # Find extreme points
            leftmost = tuple(mouth_contour[mouth_contour[:, :, 0].argmin()][0])
            rightmost = tuple(mouth_contour[mouth_contour[:, :, 0].argmax()][0])
            topmost = tuple(mouth_contour[mouth_contour[:, :, 1].argmin()][0])
            bottommost = tuple(mouth_contour[mouth_contour[:, :, 1].argmax()][0])
            
            # Calculate curve based on corner positions relative to center
            center_y = (topmost[1] + bottommost[1]) / 2
            corner_y = (leftmost[1] + rightmost[1]) / 2
            
            # Positive = smile, negative = frown
            curve = (center_y - corner_y) / h_mouth if h_mouth > 0 else 0
        else:
            curve = 0
        
        # Mouth openness (height relative to width)
        openness = h_mouth / w_mouth if w_mouth > 0 else 0.5
        
        # Mouth width relative to face
        width = w_mouth / w
        
        return {
            'curve': curve,
            'openness': openness,
            'width': width
        }
    
    def classify_emotion(self, eyebrow_data, eye_data, mouth_data):
        """Classify emotion based on facial feature analysis"""
        
        # Initialize emotion scores
        scores = {emotion: 0 for emotion in self.emotion_labels}
        
        # Analyze features for each emotion
        
        # HAPPY: upward mouth curve, slightly closed eyes (smiling)
        if mouth_data['curve'] > 0.1:
            scores['happy'] += 0.6
        if eye_data['openness'] < 0.6:  # Squinting from smiling
            scores['happy'] += 0.3
        if eyebrow_data['height'] > 0.4:  # Relaxed eyebrows
            scores['happy'] += 0.1
        
        # SAD: downward mouth curve, droopy eyes, raised inner eyebrows
        if mouth_data['curve'] < -0.1:
            scores['sad'] += 0.5
        if eye_data['openness'] < 0.5:  # Droopy eyes
            scores['sad'] += 0.3
        if eyebrow_data['angle'] > 5:  # Slightly angled eyebrows
            scores['sad'] += 0.2
        
        # ANGRY: tight mouth, narrow eyes, angled eyebrows
        if mouth_data['width'] < 0.15:  # Tight mouth
            scores['angry'] += 0.4
        if eye_data['openness'] < 0.4:  # Narrow eyes
            scores['angry'] += 0.4
        if eyebrow_data['angle'] < -5:  # Angled down eyebrows
            scores['angry'] += 0.2
        
        # SURPRISE: wide eyes, raised eyebrows, open mouth
        if eye_data['openness'] > 0.7:
            scores['surprise'] += 0.4
        if eyebrow_data['height'] < 0.3:  # Raised eyebrows
            scores['surprise'] += 0.3
        if mouth_data['openness'] > 0.6:  # Open mouth
            scores['surprise'] += 0.3
        
        # FEAR: wide eyes, raised eyebrows, tense mouth
        if eye_data['openness'] > 0.6:
            scores['fear'] += 0.3
        if eyebrow_data['height'] < 0.35:  # Raised eyebrows
            scores['fear'] += 0.3
        if mouth_data['openness'] > 0.4 and mouth_data['curve'] < 0:
            scores['fear'] += 0.4
        
        # DISGUST: wrinkled nose area, raised upper lip
        if mouth_data['curve'] > 0.05 and mouth_data['openness'] < 0.4:
            scores['disgust'] += 0.5
        if eye_data['openness'] < 0.5:  # Squinted
            scores['disgust'] += 0.3
        if eyebrow_data['angle'] < -2:
            scores['disgust'] += 0.2
        
        # NEUTRAL: balanced features
        if abs(mouth_data['curve']) < 0.05:
            scores['neutral'] += 0.3
        if 0.4 < eye_data['openness'] < 0.6:
            scores['neutral'] += 0.4
        if 0.3 < eyebrow_data['height'] < 0.5:
            scores['neutral'] += 0.3
        
        # Find the emotion with highest score
        best_emotion = max(scores, key=scores.get)
        confidence = scores[best_emotion]
        
        # Apply confidence threshold
        if confidence < 0.3:
            best_emotion = 'neutral'
            confidence = 0.5
        
        return best_emotion, min(confidence, 1.0)
    
    def detect_emotion(self, image):
        """Main emotion detection function"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_gray = gray[y:y+h, x:x+w]
            
            # Resize for consistent analysis
            face_gray = cv2.resize(face_gray, (200, 200))
            
            # Apply histogram equalization
            face_gray = cv2.equalizeHist(face_gray)
            
            # Analyze facial features
            eyebrow_data = self.analyze_eyebrows(face_gray)
            eye_data = self.analyze_eyes(face_gray)
            mouth_data = self.analyze_mouth(face_gray)
            
            # Classify emotion
            emotion, confidence = self.classify_emotion(eyebrow_data, eye_data, mouth_data)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'features': {
                    'eyebrows': eyebrow_data,
                    'eyes': eye_data,
                    'mouth': mouth_data
                }
            })
        
        return results
    
    def run_detection(self):
        """Run real-time emotion detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎭 Advanced Emotion Detection Started!")
        print("📷 Camera is running - press 'q' to quit")
        print("😊 Try different expressions:")
        print("   - 😊 Happy (smile)")
        print("   - 😢 Sad (frown)")
        print("   - 😠 Angry (frown + narrow eyes)")
        print("   - 😮 Surprised (wide eyes + open mouth)")
        print("   - 😐 Neutral (relaxed face)")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                results = self.detect_emotion(frame)
            
            # Draw results
            for result in results:
                x, y, w, h = result['bbox']
                emotion = result['emotion']
                confidence = result['confidence']
                
                # Color coding for emotions
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
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{emotion.upper()}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw status at top
                status = f"EMOTION: {emotion.upper()}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw emotion-specific indicators
                if emotion == 'happy':
                    cv2.putText(frame, "😊 HAPPY DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif emotion == 'sad':
                    cv2.putText(frame, "😢 SAD DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif emotion == 'angry':
                    cv2.putText(frame, "😠 ANGRY DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif emotion == 'surprise':
                    cv2.putText(frame, "😮 SURPRISED DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif emotion == 'fear':
                    cv2.putText(frame, "😨 FEAR DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
                elif emotion == 'disgust':
                    cv2.putText(frame, "🤢 DISGUST DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 128), 2)
                
                # Draw confidence bar
                bar_width = int(confidence * 200)
                cv2.rectangle(frame, (10, 70), (10 + bar_width, 85), color, -1)
                cv2.rectangle(frame, (10, 70), (210, 85), (255, 255, 255), 1)
                
            # Display frame
            cv2.imshow('Advanced Emotion Detection', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎯 Advanced emotion detection stopped")

def main():
    detector = AdvancedEmotionDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()
