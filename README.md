# Quizzically
Testing Simplified

This is a module for offline proctoring and analytics of examinations.
There are three main modules:

1. Facial Recognition:
    Uses openCV to detect and track faces from the webcam.
    Can collect, train on, and recognize faces.
    
    Dependencies - openCV, numpy
    
    Steps to run:
    Delete tempfile from dataset/
    Run detect.py first to store 100 each images of subjects
    Run training.py to train models for facial recognition
    Run recog.py to detect faces.
    
2. Sentiment Analysis:
    Uses openCV, tensorflow, and keras to generate predictive models for sentiments
    and display emotions of person with webcam.
    
    Dependencies - openCV, tensorflow, keras, pandas, numpy
    
    Steps to run:
    Unzip dataset.zip into folder 'dataset'
    run trainer.py to generate new model, or use pretrained model.h5
    run videoEmot.py to run sentiment analysis.
 
3. Voice-tracking:
    Records surrounding for a set period of time and generates transcript of spoken
    words for proctoring.
    
    Dependencies - pyaudio, speech-recognition, threading
    
    Steps to run:
    Run voice.py to start recording and transcription.
