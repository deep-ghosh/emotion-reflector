# Face Emotion Recognition

This project implements a real-time face emotion recognition system using a pre-trained deep learning model. The system captures video from a webcam, detects faces, preprocesses the images, and predicts emotions.

## Project Structure

```
face-emotion-recognition
├── src
│   ├── realtimedetection.py      # Main code for real-time emotion recognition
│   ├── train_model.py             # Code for training the emotion recognition model
│   └── utils
│       ├── __init__.py            # Initialization file for utils package
│       └── preprocessing.py        # Functions for preprocessing images
├── models
│   ├── facialemotionmodel.json     # Model architecture in JSON format
│   └── facialemotionmodel.h5       # Trained model weights
├── data
│   ├── train                       # Directory for training dataset
│   └── test                        # Directory for testing dataset
├── requirements.txt                # Python dependencies
├── setup.py                        # Packaging information
└── README.md                       # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-emotion-recognition
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Real-Time Emotion Detection

To run the real-time emotion detection, execute the following command:

```
python src/realtimedetection.py
```

Make sure your webcam is connected and accessible. The program will open a window displaying the video feed with detected faces and predicted emotions.

### Training the Model

To train the emotion recognition model, run:

```
python src/train_model.py
```

This script will load the training data, define the model architecture, and fit the model to the training data.

## Data

Place your training and testing datasets in the `data/train` and `data/test` directories, respectively. Ensure that the images are properly labeled for effective training.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.