import math  # Import the math module to perform mathematical operations like calculating distance


class Tracker:
    def __init__(self):
        # Dictionary to store the center positions of detected objects
        self.center_points = {}

        # Counter to keep track of the number of unique object IDs
        self.id_count = 0

    def update(self, objects_rect):
        # List to store bounding boxes along with their assigned IDs
        objects_bbs_ids = []

        # Iterate through each detected object rectangle
        for rect in objects_rect:
            x, y, w, h = rect  # Extract the x, y coordinates, width, and height of the object

            # Calculate the center point of the object
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Flag to check if the object was already detected
            same_object_detected = False

            # Iterate through existing tracked objects to check if this one is already detected
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])  # Compute the Euclidean distance

                if dist < 35:  # If the distance is small, consider it the same object
                    self.center_points[id] = (cx, cy)  # Update the object's center position
                    objects_bbs_ids.append([x, y, w, h, id])  # Append object data with the existing ID
                    same_object_detected = True  # Mark the object as detected
                    break  # Exit the loop since we found the object

            # If it's a new object (not detected before), assign a new ID
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)  # Store the new object's center position
                objects_bbs_ids.append([x, y, w, h, self.id_count])  # Append object data with a new ID
                self.id_count += 1  # Increment the object ID counter

        # Dictionary to store only active objects (removing old, unused ones)
        new_center_points = {}

        # Iterate through the currently tracked objects to update the active list
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id  # Extract the object ID
            center = self.center_points[object_id]  # Retrieve the center position of the object
            new_center_points[object_id] = center  # Add the object to the new dictionary

        # Update the main dictionary, removing objects that are no longer detected
        self.center_points = new_center_points.copy()

        # Return the list of object bounding boxes with their associated IDs
        return objects_bbs_ids
