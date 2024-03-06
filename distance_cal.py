import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load YOLO model
model = YOLO("yolov8s.pt")

# Open video file
cap = cv2.VideoCapture("D:/HAMZA_WORK/vision_eye/video.mp4")

# Get video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Create VideoWriter object to save the processed video
out = cv2.VideoWriter('visioneye-distance-calculation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

# Define the center point of the vision eye and pixels per meter
center_point = (0, h)
pixel_per_meter = 10

# Define colors for text, text background, and bounding box
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

# Main loop for processing each frame of the video
while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        # Break the loop if the video frame is empty or processing is complete
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Create Annotator object to annotate the frame
    annotator = Annotator(im0, line_width=2)

    # Perform object detection and tracking using YOLO model
    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        # Get the track IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Loop through detected objects and their track IDs
        for box, track_id in zip(boxes, track_ids):
            # Annotate bounding boxes and track IDs on the frame
            annotator.box_label(box, label=str(track_id), color=bbox_clr)
            annotator.visioneye(box, center_point)
 
            # Calculate the distance between the centroid of the bounding box and the center point
            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid
            distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter

            # Add text displaying the distance on the frame
            text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(im0, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
            cv2.putText(im0, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

    # Write the annotated frame to the output video
    out.write(im0)
    # Display the annotated frame
    cv2.imshow("visioneye-distance-calculation", im0)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and video writer objects
out.release()
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
