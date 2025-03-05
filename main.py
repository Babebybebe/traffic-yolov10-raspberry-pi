import cv2  # Import OpenCV for image processing
from ultralytics import YOLO  # Import YOLO model for object detection
import pandas as pd  # Import Pandas for handling detection results
import cvzone  # Import cvzone for easier OpenCV operations

# Load the YOLO model using a TensorFlow Lite version for better efficiency
model = YOLO('best_float32.tflite')  

# Open a video file for processing
cap = cv2.VideoCapture('vdo1.mp4')

# Read the class names from a text file (each class is in a new line)
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")  # Convert file content into a list of class names

count = 0  # Counter to track frames

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if frame is None:  # If the video ends, break the loop
        break
    
    count += 1
    if count % 3 != 0:  # Skip frames to improve processing speed (process every 3rd frame)
        continue
    
    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistent processing

    # Run YOLO model on the frame, setting the image size to 240px
    results = model(frame, imgsz=240)
    
    a = results[0].boxes.data  # Extract detection data (bounding boxes, confidence, class)
    px = pd.DataFrame(a).astype("float")  # Convert results into a Pandas DataFrame for easy handling
    
    car = []  # List to store detected cars (not used further in this script)
    motorcycle = []  # List to store detected motorcycles (not used further in this script)

    # Iterate through each detected object
    for index, row in px.iterrows():
        x1 = int(row[0])  # X-coordinate of the top-left corner
        y1 = int(row[1])  # Y-coordinate of the top-left corner
        x2 = int(row[2])  # X-coordinate of the bottom-right corner
        y2 = int(row[3])  # Y-coordinate of the bottom-right corner
        
        d = int(row[5])  # Class ID of detected object
        c = class_list[d]  # Get the class name from the list
        
        # Draw a label with the class name on the frame
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        
        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle with 2px thickness
  
    cv2.imshow("FRAME", frame)  # Display the processed frame
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
