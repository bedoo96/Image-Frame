import os
import cv2
import argparse

# Load the pre-trained Haar Cascade classifier for full-body pedestrians
cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", type=str, required=True, help="Input video file path")
args = vars(ap.parse_args())

# Create a directory for storing extracted frames
output_dir = 'test_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
video_path = args["video"]
cap = cv2.VideoCapture(video_path)

# Initialize frame counter
current_frame = 0

# Process the video frame by frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Break the loop if the video has ended
    if not ret:
        break
    
    # Resize the frame to a smaller size
    frame = cv2.resize(frame, (800, 600))
    
    # Detect full-body pedestrians in the frame
    regions = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Write the extracted frames to disk
    for (x, y, w, h) in regions:
        if w > 60 and h > 60:
            # Extract the region containing the pedestrian
            pedestrian_img = frame[y:y+h, x:x+w]
            
            # Save the extracted image to disk
            output_file = os.path.join(output_dir, f'frame{current_frame}.png')
            cv2.imwrite(output_file, pedestrian_img)
            
            # Increment the frame counter
            current_frame += 1

# Release the video capture object and close all windows
# Release the video capture object
cap.release()

  # Replace 'Window Name' with the actual window name(s) used in your code

