"""
Quick Model Training Solution for Face Emotion Recognition
This script provides a simple training approach using synthetic data and data augmentation
"""

import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

class QuickTrainer:
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = None
        
    def create_improved_model(self):
        """Create an improved CNN model for emotion recognition"""
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional block
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third convolutional block
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_synthetic_data(self, samples_per_emotion=100):
        """Generate synthetic training data with emotion-specific patterns"""
        print("🎭 Generating synthetic training data...")
        
        X_train = []
        y_train = []
        
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            print(f"   Generating {samples_per_emotion} samples for {emotion}...")
            
            for i in range(samples_per_emotion):
                # Create base face template
                img = np.ones((48, 48), dtype=np.uint8) * 128
                
                # Add emotion-specific patterns
                if emotion == 'happy':
                    # Upward curved mouth
                    cv2.ellipse(img, (24, 35), (8, 4), 0, 0, 180, 255, -1)
                    # Slightly closed eyes (smiling)
                    cv2.ellipse(img, (16, 20), (4, 2), 0, 0, 180, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 2), 0, 0, 180, 0, -1)
                    
                elif emotion == 'sad':
                    # Downward curved mouth
                    cv2.ellipse(img, (24, 38), (8, 4), 0, 180, 360, 0, -1)
                    # Droopy eyes
                    cv2.ellipse(img, (16, 20), (4, 3), 0, 0, 180, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 3), 0, 0, 180, 0, -1)
                    # Droopy eyebrows
                    cv2.line(img, (12, 15), (20, 17), 0, 2)
                    cv2.line(img, (28, 17), (36, 15), 0, 2)
                    
                elif emotion == 'angry':
                    # Straight/downward mouth
                    cv2.line(img, (18, 36), (30, 36), 0, 2)
                    # Narrowed eyes
                    cv2.ellipse(img, (16, 20), (4, 1), 0, 0, 180, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 1), 0, 0, 180, 0, -1)
                    # Angled eyebrows
                    cv2.line(img, (10, 15), (20, 18), 0, 2)
                    cv2.line(img, (28, 18), (38, 15), 0, 2)
                    
                elif emotion == 'surprise':
                    # Open mouth
                    cv2.ellipse(img, (24, 36), (4, 6), 0, 0, 360, 0, -1)
                    # Wide eyes
                    cv2.ellipse(img, (16, 20), (4, 4), 0, 0, 360, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 4), 0, 0, 360, 0, -1)
                    # Raised eyebrows
                    cv2.line(img, (12, 12), (20, 14), 0, 2)
                    cv2.line(img, (28, 14), (36, 12), 0, 2)
                    
                elif emotion == 'fear':
                    # Slightly open mouth
                    cv2.ellipse(img, (24, 36), (3, 4), 0, 0, 360, 0, -1)
                    # Wide eyes
                    cv2.ellipse(img, (16, 20), (4, 4), 0, 0, 360, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 4), 0, 0, 360, 0, -1)
                    # Raised eyebrows
                    cv2.line(img, (12, 13), (20, 15), 0, 2)
                    cv2.line(img, (28, 15), (36, 13), 0, 2)
                    
                elif emotion == 'disgust':
                    # Raised upper lip
                    cv2.line(img, (20, 32), (28, 32), 0, 2)
                    cv2.line(img, (18, 36), (30, 36), 0, 2)
                    # Squinted eyes
                    cv2.ellipse(img, (16, 20), (4, 1), 0, 0, 180, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 1), 0, 0, 180, 0, -1)
                    
                else:  # neutral
                    # Neutral mouth
                    cv2.line(img, (20, 36), (28, 36), 128, 1)
                    # Normal eyes
                    cv2.ellipse(img, (16, 20), (4, 2), 0, 0, 180, 0, -1)
                    cv2.ellipse(img, (32, 20), (4, 2), 0, 0, 180, 0, -1)
                
                # Add noise and variations
                noise = np.random.normal(0, 10, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
                # Random transformations
                if np.random.random() > 0.5:
                    # Random rotation
                    angle = np.random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((24, 24), angle, 1)
                    img = cv2.warpAffine(img, M, (48, 48))
                
                # Normalize
                img = img.astype('float32') / 255.0
                
                X_train.append(img)
                y_train.append(emotion_idx)
        
        X_train = np.array(X_train).reshape(-1, 48, 48, 1)
        y_train = to_categorical(y_train, 7)
        
        print(f"✅ Generated {len(X_train)} training samples")
        return X_train, y_train
    
    def train_model(self, epochs=20):
        """Train the emotion recognition model"""
        print("🚀 Starting model training...")
        
        # Generate synthetic data
        X_train, y_train = self.generate_synthetic_data(samples_per_emotion=200)
        
        # Create and compile model
        self.model = self.create_improved_model()
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        # Train the model
        print("🎯 Training model...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            verbose=1,
            validation_data=(X_train, y_train),
            steps_per_epoch=len(X_train) // 32
        )
        
        print("✅ Model training completed!")
        return history
    
    def save_trained_model(self):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return
        
        try:
            # Save model architecture
            model_json = self.model.to_json()
            with open("facialemotionmodel.json", "w") as json_file:
                json_file.write(model_json)
            
            # Save model weights
            self.model.save_weights("facialemotionmodel.weights.h5")
            
            print("✅ Trained model saved successfully!")
            print("📁 Files created:")
            print("   - facialemotionmodel.json (model architecture)")
            print("   - facialemotionmodel.weights.h5 (trained weights)")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("❌ No model to evaluate. Train the model first.")
            return
        
        # Generate test data
        X_test, y_test = self.generate_synthetic_data(samples_per_emotion=50)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"📊 Model Evaluation:")
        print(f"   Test Loss: {loss:.4f}")
        print(f"   Test Accuracy: {accuracy:.4f}")
        
        return accuracy

def main():
    print("🎭 Quick Face Emotion Recognition Training")
    print("=" * 50)
    
    trainer = QuickTrainer()
    
    # Train the model
    trainer.train_model(epochs=30)
    
    # Evaluate the model
    trainer.evaluate_model()
    
    # Save the trained model
    trainer.save_trained_model()
    
    print("\n🎉 Training complete! You can now use the trained model with:")
    print("   python enhanced_emotion_detector.py")
    print("   python realtimedetection.py")

if __name__ == "__main__":
    main()
