# distance_calculation-Using-YOLO_v8 | Vision Eye


## Overview
This Python script demonstrates how to calculate the distance of objects from a fixed point in a video stream using the YOLO object detection model. The script loads a pre-trained YOLO model and processes each frame of a video to detect objects, calculate their distance from a designated center point, and annotate the frames with the distance information.

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO v8 library

## Usage
1. Ensure you have Python 3.x installed on your system.
2. Install OpenCV and Ultralytics YOLO library if not already installed:
   ```bash
   pip install opencv-python-headless
   pip install ultralytics

* Adjust the video_path variable in the script to point to the location of your input video file.
* Set the appropriate parameters such as the center point, pixel per meter ratio, and colors for annotation.
* Run the script python visioneye_distance_calculation.py


##Functionality

* Load YOLO Model: The script loads the YOLO object detection model from the Ultralytics library.
* Open Video File: The input video file is opened using OpenCV's VideoCapture.
* Process Frames: Each frame of the video is processed sequentially.
* Object detection and tracking are performed using the YOLO model.
* Bounding boxes and track IDs are annotated on the frames.
* The distance of objects from the designated center point is calculated and displayed on the frames.
* Save Processed Video: The annotated frames are written to an output video file using OpenCV's VideoWriter.
* Display Results: The processed video with annotations is displayed in real-time using OpenCV's imshow.


##Example
An example usage of the script is provided in the code. Replace the video_path variable with the path to your input video file, and adjust other parameters as necessary.

##Notes
*  Fine-tune the parameters such as the center point and pixel per meter ratio according to your specific requirements and video characteristics.
*  This script utilizes the YOLO object detection model for detecting objects in the video frames. Ensure that the model performs adequately for your application.
*  You may need to customize the code further for advanced features or specific use cases.
