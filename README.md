# YOLO Object Detection 

## Overview
This project demonstrates the implementation of YOLO (You Only Look Once), a state-of-the-art, real-time object detection system. The primary aim of the project is to detect various objects within an input image using a pre-trained YOLO model. The detection is carried out with the help of OpenCV's deep learning module.

## Table of Contents
- [Overview](#overview)
- [Model Used](#model-used)
- [Output](#output)
- [Weights Used](#weights-used)
- [Objects That Can Be Detected](#objects-that-can-be-detected)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Additional Notes](#additional-notes)

## Model Used
The model used in this project is YOLOv3 (You Only Look Once version 3). YOLOv3 is known for its speed and accuracy in object detection tasks. Unlike traditional methods, YOLO applies a single neural network to the full image, which divides the image into regions and predicts bounding boxes and probabilities for each region.

## Output

The output of the YOLO model includes:
- **Bounding Boxes**: Rectangular boxes around detected objects.
- **Confidence Scores**: Probability scores indicating the likelihood that the detected object belongs to a particular class.
- **Class Labels**: Identified object classes such as 'person', 'car', 'bicycle', etc.

<img src="https://github.com/sumithanwate3/YOLO_Object-Detection/assets/96422074/d02cd464-1bb5-47a9-a105-a8f47c9dac83" alt="YOLO Image 1" style="width: 100%; max-width: 800px;">


## Weights Used
The project uses pre-trained weights provided by the YOLOv3 model. These weights are stored in the file `yolov3.weights`. The model configuration is defined in `yolov3.cfg`.

## Objects That Can Be Detected
The YOLOv3 model is trained on the COCO dataset, which includes 80 object categories. Some of the objects that can be detected are:
- People
- Vehicles (cars, buses, trucks, motorcycles)
- Animals (dogs, cats, birds)
- Everyday objects (bottles, chairs, sofas, potted plants)
- Electronic items (laptops, cell phones, remote controls)
- Traffic signs (stop signs, fire hydrants)

For a complete list of detectable objects, refer to the file `COCO names.names`.

## Dependencies
The following dependencies are required to run this project:
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the dependencies using the following command:
```sh
pip install opencv-python numpy matplotlib
```

## Usage
1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Place the Model Files**:
   Ensure `yolov3.weights`, `yolov3.cfg`, and `COCO names.names` are placed in the project directory.

3. **Run the Script**:
   Modify the `args` dictionary in the script to specify the path to your input image. Run the script using:
   ```sh
   python yolo_detection.py
   ```

4. **View the Results**:
   The output image with detected objects will be displayed using Matplotlib.

## Results
After running the script, the detected objects will be highlighted with bounding boxes in the output image. Each bounding box will be labeled with the corresponding class name and confidence score. The following example shows the detection results on an input image:

![Detected Objects](output_image.jpg)

## Additional Notes
- **Performance**: YOLOv3 is highly efficient and capable of real-time object detection. However, the performance may vary based on the hardware capabilities.
- **Confidence Threshold**: The confidence threshold (`args['confidence']`) can be adjusted to filter out weak detections. Increasing this value will result in fewer, but more accurate detections.
- **Non-Maxima Suppression (NMS)**: The NMS threshold (`args['threshold']`) helps in removing multiple detections of the same object. A lower threshold will result in fewer overlapping boxes.

By following this guide, you should be able to successfully run the YOLO object detection on your images and interpret the results effectively. For any further queries or issues, please refer to the official YOLO documentation and the OpenCV library reference.
