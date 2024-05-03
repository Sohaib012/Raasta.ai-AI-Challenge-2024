import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Read the image
#image = cv2.imread("D:/Adaptive-Traffic-Signal-Control-System-master/Adaptive-Traffic-Signal-Control-System-master/traffic.jpeg")
#cap = cv2.VideoCapture('Traffic cars passing in road with asphalt with cracks seen from above 36990287 Stock Video at Vecteezy.mp4')
cap = cv2.VideoCapture('traffic.mp4')


def visualize_lanes(image, lane_rois):
    img_copy = image.copy()

    # Draw polygons for each lane
    for lane_roi in lane_rois:
        pts = np.array(lane_roi, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img_copy, [pts], True, (0,255,0), 2)

    return img_copy

def count_vehicles_in_lanes(results, image, lane_rois):
    img_copy = image.copy()
    #vehicle_ids = [[] for _ in range(len(lane_rois))] 

    # Iterate through the results
    for result in results:
        for box in result.boxes:       
            #print(result)
            center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

            # Check if the center of the bounding box lies within the lane polygon
            for i, lane_roi in enumerate(lane_rois):
                lane_polygon = np.array(lane_roi, np.int32)
                is_inside = cv2.pointPolygonTest(lane_polygon, (center_x, center_y), False)
                if is_inside >= 0:
                    if box.id[0] not in already_counted:
                        lane_vehicle_counts[i] += 1
                        already_counted[i].append(box.id[0])
                #print(already_counted)
            # Draw bounding box and label on the image
            cv2.rectangle(img_copy, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                        (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            # Draw the center of the bounding box as a small circle
            cv2.circle(img_copy, (center_x, center_y), 2, (0, 255, 0), -1)
            cv2.putText(img_copy, f"ID: {int(box.id[0])} {result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)     
            
            # Visualize lanes
            img_copy = visualize_lanes(img_copy, lane_rois)

        # Print the vehicle counts for each lane
        for i in range(len(lane_rois)):
            cv2.putText(img_copy, f"Lane {i + 1}: {lane_vehicle_counts[i]}",
                        (1, 30 * (i + 1)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # for i in range(len(lane_rois) - 1):
        #     lane_vehicle_counts[i] = lane_vehicle_counts[i] - lane_vehicle_counts[i+1] 

        return img_copy, lane_vehicle_counts


while True:
    _, image = cap.read()
    img_height, img_width, _ = image.shape

    # Define the ROIs for each lane (adjust these coordinates according to your image)
    lane_rois = [
        ((387, 367), (457, 367), (456, img_height), (302, img_height)),
        ((457, 367), (524, 367), (609, img_height), (456, img_height)),
        (((524, 367)), (650, 367), (897, img_height), ((609, img_height))) 
    ]

    # Define initial green signal times for each lane
    green_signal_times = [10] * len(lane_rois)   # Initial green signal time for each lane in seconds

    lane_vehicle_counts = [0] * len(lane_rois) 
    already_counted = [[] for _ in range(len(lane_rois))] 

    # Perform object detection
    results = model.track(image, persist=True)

    # Count vehicles in each lane
    result_img, vehicle_counts = count_vehicles_in_lanes(results, image, lane_rois)

    # Update green signal times based on vehicle counts
    for i, count in enumerate(vehicle_counts):
        # Adjust green signal time proportional to the number of vehicles in the lane
        green_signal_times[i] = max(5, min(30, 10 + count))  # Example adjustment formula

    # Display the image with lanes, bounding boxes, and labels
    cv2.imshow("Demo", result_img)
    #cv2.imshow("Image1", lane_visualized_image)
    cv2.imwrite("result.jpeg", result_img)
    #cv2.waitKey(0)

    # Print the adjusted green signal times for each lane
    gst = max(green_signal_times)
    print(f"Green Signal Time: {gst} seconds")
 

    # Print the vehicle counts for each lane
    # for i, count in enumerate(vehicle_counts):
    #     print(f"Lane {i+1}: {count} vehicles")



    #cv2.imshow("Frame",image)
    key = cv2.waitKey(0)
    if key ==27:
        break
for i, count in enumerate(vehicle_counts):
    print(f"Lane {i+1}: {count} vehicles")

cap.release() 
cv2.destroyAllWindows()
 
# Define the ROIs for each lane (adjust these coordinates according to your image)
# lane_rois = [
#     ((0, img_height // 2), (img_width, img_height)),     # Top lane
#     ((img_width // 3, img_height // 2), (img_width * 2, img_height)),  # Middle lane
#     (((img_width // 3)  * 2, img_height // 2), (img_width, img_height))        # Bottom lane
# ]

