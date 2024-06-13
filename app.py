import cv2
import torch
import numpy as np
import time
import os
import simpleaudio as sa
import threading


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up alarm sound
def play_alarm():
    wave_obj = sa.WaveObject.from_wave_file('Audio/Alarm.wav')
    play_obj = wave_obj.play()
    #play_obj.wait_done()

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform detection
    results = model(frame)

    # Parse results
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # Loop through detections and check for humans or animals
    for label, cord in zip(labels, cords):
        x1, y1, x2, y2, conf = cord
        if conf > 0.5:
            if int(label) == 0:  # Label 0 corresponds to 'person'
                play_alarm()
                print("Human detected")
            elif int(label) in [15, 16, 17]:  # Labels for cat, dog, horse
                print("Animal detected")

    # Display the resulting frame
    cv2.imshow('frame', np.squeeze(results.render()))

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
