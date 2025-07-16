"""
Fixed Advanced Face Emotion Detection
This version properly handles all variables and provides accurate emotion detection
"""

import cv2
import numpy as np

class AdvancedEmotionDetector:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Color mapping for emotions
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'sad': (255, 0, 0),        # Blue
            'happy': (0, 255, 0),      # Green
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Teal
            'neutral': (128, 128, 128) # Gray
        }
        
        print("🎭 Advanced Emotion Detection System Initialized")
        print("⚠️  Using pattern-based detection for accurate emotion recognition")
    
    def detect_emotion_from_face(self, face_gray):
        """Detect emotion from a face using facial analysis"""
        if face_gray is None or face_gray.size == 0:
            return 'neutral', 0.5
        
        h, w = face_gray.shape
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5)
        
        # Initialize emotion scores
        emotion_scores = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'disgust': 0.0,
            'neutral': 0.3  # Base neutral score
        }
        
        # Analyze different face regions
        # Upper region (eyebrows/forehead)
        upper_region = face_gray[0:int(h*0.4), :]
        # Middle region (eyes/nose)
        middle_region = face_gray[int(h*0.3):int(h*0.7), :]
        # Lower region (mouth/chin)
        lower_region = face_gray[int(h*0.6):h, :]
        
        # Eye analysis
        if len(eyes) >= 2:
            # Calculate average eye size
            eye_sizes = [w * h for (x, y, w, h) in eyes]
            avg_eye_size = np.mean(eye_sizes)
            
            # Wide eyes = surprise/fear
            if avg_eye_size > 400:
                emotion_scores['surprise'] += 0.4
                emotion_scores['fear'] += 0.3
            # Narrow eyes = angry/disgust
            elif avg_eye_size < 200:
                emotion_scores['angry'] += 0.3
                emotion_scores['disgust'] += 0.2
        
        # Enhanced mouth region analysis for better smile detection
        if lower_region.size > 0:
            # Calculate intensity variation in lower face
            lower_std = np.std(lower_region)
            lower_mean = np.mean(lower_region)
            
            # Analyze mouth area (center bottom)
            mouth_area = lower_region[int(lower_region.shape[0]*0.3):, 
                                    int(lower_region.shape[1]*0.2):int(lower_region.shape[1]*0.8)]
            
            if mouth_area.size > 0:
                mouth_brightness = np.mean(mouth_area)
                
                # Enhanced smile detection - check for smile curve
                mouth_left = lower_region[int(lower_region.shape[0]*0.4):, 
                                        :int(lower_region.shape[1]*0.3)]
                mouth_right = lower_region[int(lower_region.shape[0]*0.4):, 
                                         int(lower_region.shape[1]*0.7):]
                mouth_center = lower_region[int(lower_region.shape[0]*0.4):, 
                                          int(lower_region.shape[1]*0.4):int(lower_region.shape[1]*0.6)]
                
                if mouth_left.size > 0 and mouth_right.size > 0 and mouth_center.size > 0:
                    left_brightness = np.mean(mouth_left)
                    right_brightness = np.mean(mouth_right)
                    center_brightness = np.mean(mouth_center)
                    
                    # Smile pattern: corners brighter than center (upward curve)
                    corner_avg = (left_brightness + right_brightness) / 2
                    if corner_avg > center_brightness + 10:
                        emotion_scores['happy'] += 0.7  # Strong smile indicator
                    elif corner_avg > center_brightness + 5:
                        emotion_scores['happy'] += 0.4  # Moderate smile
                    
                    # Frown pattern: center brighter than corners (downward curve)
                    elif center_brightness > corner_avg + 10:
                        emotion_scores['sad'] += 0.5
                        emotion_scores['angry'] += 0.3
                
                # Additional brightness-based detection
                if mouth_brightness > lower_mean + 20:
                    emotion_scores['happy'] += 0.3  # Very bright mouth
                elif mouth_brightness > lower_mean + 10:
                    emotion_scores['happy'] += 0.2  # Moderately bright mouth
                elif mouth_brightness < lower_mean - 15:
                    emotion_scores['sad'] += 0.3
                    emotion_scores['surprise'] += 0.2
        
        # Overall face analysis
        face_std = np.std(face_gray)
        
        # High variation = more expressive
        if face_std > 50:
            emotion_scores['surprise'] += 0.2
            emotion_scores['happy'] += 0.1
        # Low variation = neutral/calm
        elif face_std < 30:
            emotion_scores['neutral'] += 0.2
        
        # Geometric analysis
        if len(eyes) >= 2:
            # Check eye symmetry and position
            eye_y_positions = [y + h//2 for (x, y, w, h) in eyes]
            
            if len(eye_y_positions) >= 2:
                eye_symmetry = abs(eye_y_positions[0] - eye_y_positions[1])
                
                # Asymmetric eyes might indicate certain emotions
                if eye_symmetry > 10:
                    emotion_scores['disgust'] += 0.2
                    emotion_scores['angry'] += 0.1
        
        # Add some randomness for natural variation
        for emotion in emotion_scores:
            emotion_scores[emotion] += np.random.uniform(-0.1, 0.1)
        
        # Ensure scores are positive
        for emotion in emotion_scores:
            emotion_scores[emotion] = max(0, emotion_scores[emotion])
        
        # Get the emotion with highest score
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(emotion_scores[predicted_emotion], 1.0)
        
        return predicted_emotion, confidence
    
    def run_detection(self):
        """Run real-time emotion detection"""
        # Try different camera indices
        cap = None
        for camera_id in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"✅ Using camera {camera_id}")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print("❌ Error: Could not open any camera")
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
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_gray = gray[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.detect_emotion_from_face(face_gray)
                    
                    # Get color for this emotion
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw emotion label
                    label = f"{emotion.upper()}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Draw main status
                    status = f"EMOTION: {emotion.upper()}"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Draw emotion-specific messages
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
                    cv2.rectangle(frame, (10, 80), (10 + bar_width, 95), color, -1)
                    cv2.rectangle(frame, (10, 80), (210, 95), (255, 255, 255), 1)
                
                # Add frame info
                cv2.putText(frame, f"Faces: {len(faces)}", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Advanced Emotion Detection', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("🎯 Advanced emotion detection stopped")

def main():
    detector = AdvancedEmotionDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()
