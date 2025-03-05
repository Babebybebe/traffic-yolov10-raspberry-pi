import cv2  # OpenCV for video processing
import time  # Time module (not used in the script, but can be useful for debugging)

# Counter for saved frames
cpt = 0  

# Maximum frames to capture
maxFrames = 40  

# Open the video file (add the correct file path)
cap = cv2.VideoCapture('video.mp4')  

count = 0  # Counter for frame skipping

# Check if the video file is successfully opened
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

while cpt < maxFrames:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is returned (end of video), break the loop
        break

    count += 1
    if count % 3 != 0:  # Skip 2 out of every 3 frames for efficiency
        continue

    frame = cv2.resize(frame, (1020, 600))  # Resize frame to 1020x600

    cv2.imshow("Test Window", frame)  # Display the frame

    # Save frame as an image file (ensure a valid path)
    cv2.imwrite(f"img_{cpt}.jpg", frame)  
    cpt += 1  # Increment saved frame count

    # Press 'ESC' key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
