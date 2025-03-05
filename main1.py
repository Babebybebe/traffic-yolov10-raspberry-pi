import cv2  # OpenCV for image processing
from ultralytics import YOLO  # YOLO model for object detection
import pandas as pd  # Pandas for handling detection results
import cvzone  # Cvzone for easier OpenCV operations
from tracker import Tracker  # Import custom tracking class

# Load the YOLO model (TensorFlow Lite version for efficiency)
model = YOLO('best_float32.tflite')  

# Open the video file for processing
cap = cv2.VideoCapture('vdo1.mp4')

# Load class names from a text file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")  # Convert the text file content into a list of class names

# Initialize object trackers for cars and motorcycles
tracker = Tracker()
tracker1 = Tracker()

count = 0  # Frame counter

# Define the Y-coordinate of the counting line
cy1 = 226

# Offset value for detecting objects crossing the line
offset = 10

# Lists to store counted object IDs
car_counter = []
motorcycle_counter = []    

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if frame is None:  # If the video ends, exit the loop
        break
    
    count += 1
    if count % 3 != 0:  # Process every 3rd frame to improve performance
        continue

    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistent processing

    # Run YOLO model on the frame (with image size 240px)
    results = model(frame, imgsz=240)
    
    a = results[0].boxes.data  # Extract detection results
    px = pd.DataFrame(a).astype("float")  # Convert results into a Pandas DataFrame
    
    car = []  # List to store detected car bounding boxes
    motorcycle = []  # List to store detected motorcycle bounding boxes

    # Iterate through each detected object
    for index, row in px.iterrows():
        x1 = int(row[0])  # X-coordinate of the top-left corner
        y1 = int(row[1])  # Y-coordinate of the top-left corner
        x2 = int(row[2])  # X-coordinate of the bottom-right corner
        y2 = int(row[3])  # Y-coordinate of the bottom-right corner
        
        d = int(row[5])  # Class ID of the detected object
        c = class_list[d]  # Get the class name

        # Separate cars and motorcycles into their respective lists
        if 'car' in c:
            car.append([x1, y1, x2, y2])

        if 'motorcycle' in c:
            motorcycle.append([x1, y1, x2, y2])

    # Update object trackers with detected bounding boxes
    bbox_idx = tracker.update(car)  # Track cars
    bbox_idx1 = tracker1.update(motorcycle)  # Track motorcycles
   
    # Process tracked car objects
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2  # Calculate the center X-coordinate
        cy = int(y3 + y4) // 2  # Calculate the center Y-coordinate

        # Check if the object crosses the counting line
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # Draw a small circle at the center
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)  # Display the object ID
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)  # Draw a bounding box

            # Ensure each car is counted only once
            if car_counter.count(id) == 0:
                car_counter.append(id)

    # Process tracked motorcycle objects
    for bbox1 in bbox_idx1:
        x5, y5, x6, y6, id1 = bbox1
        cx3 = int(x5 + x6) // 2  # Calculate the center X-coordinate
        cy3 = int(y5 + y6) // 2  # Calculate the center Y-coordinate

        # Check if the motorcycle crosses the counting line
        if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
            cv2.circle(frame, (cx3, cy3), 4, (255, 0, 255), -1)  # Draw a small circle at the center
            cvzone.putTextRect(frame, f'{id1}', (x5, y5), 1, 1)  # Display the object ID
            cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 0), 2)  # Draw a bounding box

            # Ensure each motorcycle is counted only once
            if motorcycle_counter.count(id1) == 0:
                motorcycle_counter.append(id1)
           
# Display car and motorcycle counters on the screen
    ccounter = len(car_counter)
    mcounter = len(motorcycle_counter)

    # Draw the counting line
    cv2.line(frame, (236, 226), (1018, 226), (0, 0, 255), 1)  # Red line for counting objects

    # Display the car count on the frame
    cvzone.putTextRect(frame, f'Car Counter: {ccounter}', (60, 50), 1, 1)

    # Display the motorcycle count on the frame
    cvzone.putTextRect(frame, f'Motorcycle Counter: {mcounter}', (60, 150), 1, 1)

    # Show the processed frame
    cv2.imshow("FRAME", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
