"""
Simple Camera Test and Emotion Detection Restart Script
This script tests different camera configurations and restarts the emotion detector
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test different camera configurations"""
    print("🎥 Testing Camera Access...")
    
    # Test different camera indices
    for camera_id in [0, 1, 2]:
        print(f"   Testing camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {camera_id} is working!")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return camera_id
            else:
                print(f"❌ Camera {camera_id} opened but can't read frames")
        else:
            print(f"❌ Camera {camera_id} failed to open")
        
        cap.release()
    
    print("❌ No working camera found")
    return None

def restart_emotion_detector():
    """Restart the emotion detection with proper camera handling"""
    print("\n🎭 Restarting Emotion Detection System...")
    
    # Test camera first
    camera_id = test_camera()
    if camera_id is None:
        print("❌ Cannot start emotion detection - no camera available")
        return
    
    # Import required modules
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if face_cascade.empty() or eye_cascade.empty():
            print("❌ Error: Could not load Haar cascade files")
            return
            
    except Exception as e:
        print(f"❌ Error loading cascades: {e}")
        return
    
    # Initialize camera with the working ID
    cap = cv2.VideoCapture(camera_id)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}")
        return
    
    print(f"✅ Camera {camera_id} initialized successfully")
    print("🎭 Emotion Detection Started!")
    print("📷 Camera is running - press 'q' to quit")
    print("\n😊 Try different expressions:")
    print("   - 😊 Happy (smile)")
    print("   - 😢 Sad (frown)")
    print("   - 😠 Angry (frown + narrow eyes)")
    print("   - 😮 Surprised (wide eyes + open mouth)")
    print("   - 😐 Neutral (relaxed face)")
    
    # Emotion detection parameters
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_colors = {
        'angry': (0, 0, 255),      # Red
        'sad': (255, 0, 0),        # Blue
        'happy': (0, 255, 0),      # Green
        'surprise': (0, 255, 255), # Yellow
        'fear': (128, 0, 128),     # Purple
        'disgust': (0, 128, 128),  # Teal
        'neutral': (128, 128, 128) # Gray
    }
    
    frame_count = 0
    last_emotion = 'neutral'
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                face_roi = frame[y:y+h, x:x+w]
                
                # Simple emotion detection based on facial features
                emotion = detect_emotion_simple(face_gray, eye_cascade)
                
                # Smooth emotion changes
                if frame_count % 10 == 0:  # Update emotion every 10 frames
                    last_emotion = emotion
                
                # Get color for emotion
                color = emotion_colors.get(last_emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{last_emotion.upper()}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw status
                status = f"EMOTION: {last_emotion.upper()}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw emotion-specific indicators
                if last_emotion == 'happy':
                    cv2.putText(frame, "😊 HAPPY DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif last_emotion == 'sad':
                    cv2.putText(frame, "😢 SAD DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif last_emotion == 'angry':
                    cv2.putText(frame, "😠 ANGRY DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif last_emotion == 'surprise':
                    cv2.putText(frame, "😮 SURPRISED DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Restarted Emotion Detection', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎯 Emotion detection stopped")

def detect_emotion_simple(face_gray, eye_cascade):
    """Simple emotion detection based on facial geometry"""
    h, w = face_gray.shape
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
    
    # Analyze mouth region
    mouth_region = face_gray[int(h*0.6):h, int(w*0.2):int(w*0.8)]
    
    # Calculate brightness in different regions
    upper_face = np.mean(face_gray[0:int(h*0.5), :])
    lower_face = np.mean(face_gray[int(h*0.5):h, :])
    mouth_brightness = np.mean(mouth_region)
    
    # Simple heuristics for emotion detection
    if len(eyes) >= 2:
        # Calculate eye positions
        eye_y_positions = [eye[1] + eye[3]//2 for eye in eyes]
        avg_eye_y = np.mean(eye_y_positions)
        
        # Check if eyes are squinted (happy) or wide (surprise)
        eye_sizes = [eye[2] * eye[3] for eye in eyes]
        avg_eye_size = np.mean(eye_sizes)
        
        # Emotion detection logic
        if mouth_brightness < lower_face * 0.9:  # Dark mouth area (open/frown)
            if avg_eye_size > 200:  # Wide eyes
                return 'surprise'
            elif avg_eye_y < h * 0.4:  # Eyes high (raised eyebrows)
                return 'fear'
            else:
                return 'sad'
        elif mouth_brightness > lower_face * 1.1:  # Bright mouth area (smile)
            if avg_eye_size < 150:  # Squinted eyes
                return 'happy'
            else:
                return 'happy'
        elif avg_eye_size < 100:  # Very narrow eyes
            return 'angry'
        else:
            return 'neutral'
    else:
        # No eyes detected, use mouth region only
        if mouth_brightness < lower_face * 0.9:
            return 'sad'
        elif mouth_brightness > lower_face * 1.1:
            return 'happy'
        else:
            return 'neutral'

def main():
    print("🎭 Face Emotion Recognition - Restart Script")
    print("=" * 50)
    
    restart_emotion_detector()

if __name__ == "__main__":
    main()
