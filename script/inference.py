"""
Real-Time YOLOv8 Object Detection/Segmentation on Camera with OpenCV

This script uses a YOLOv8 model to perform real-time object detection or segmentation on a camera feed.
It displays the video feed with bounding boxes or segmentation masks drawn on detected objects using OpenCV.
Detected objects are classified as 'Bundle' (class 0) or 'Single-Item' (class 1).

Usage:
    python3 real_time_yolo.py <model_path> [--confidence <value>] [--epsilon <value>] [--resize-height <height>] [--resize-width <width>]

Arguments:
    model_path: Path to the YOLOv8 .pt model file.
    --confidence: Confidence threshold for detection (default: 0.5).
    --epsilon: Optional value for polygon simplification (default: 0.01).
    --resize-height: Optional height to resize frames before processing.
    --resize-width: Optional width to resize frames before processing.

Example:
    python3 real_time_yolo.py /path/to/model.pt --confidence 0.5 --epsilon 0.02 --resize-height 640 --resize-width 480
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

class RealTimeYOLO:
    def __init__(self, model_path, confidence=0.5, epsilon=0.01, resize_height=None, resize_width=None):
        self.model_path_ = model_path
        self.confidence_ = confidence
        self.epsilon_ = epsilon
        self.resize_height_ = resize_height
        self.resize_width_ = resize_width

        # Load YOLO model
        self.model_ = YOLO(self.model_path_, task='segment' if 'segment' in self.model_path_ else 'detect')
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize video capture (default webcam)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

    def draw_masks(self, image, masks, classes, confidences):
        for mask, cls, conf in zip(masks, classes, confidences):
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            label = f"{'sample' if cls == 0 else 'other item'} {conf:.2f}"
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            simplified_contours = [self.simplify_contour(contour) for contour in contours]
            cv2.drawContours(image, simplified_contours, -1, color, 2)
            # Add label near the contour
            if simplified_contours:
                x, y = simplified_contours[0][0][0]
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def draw_bounding_boxes(self, image, boxes, classes, confidences):
        for box, cls, conf in zip(boxes, classes, confidences):
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            label = f"{'sample' if cls == 0 else 'other item'} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def simplify_contour(self, contour):
        epsilon = self.epsilon_ * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Optionally resize the frame
                if self.resize_height_ is not None and self.resize_width_ is not None:
                    frame = cv2.resize(frame, (self.resize_width_, self.resize_height_))

                # Perform inference
                results = self.model_.predict(
                    source=frame,
                    verbose=False,
                    stream=False,
                    conf=self.confidence_,
                    device=self.device_
                )[0].cpu()

                # Process results
                if hasattr(results, 'masks') and results.masks:
                    frame = self.draw_masks(frame, results.masks.data, results.boxes.cls, results.boxes.conf)
                elif hasattr(results, 'boxes') and results.boxes:
                    frame = self.draw_bounding_boxes(frame, results.boxes.xyxy, results.boxes.cls, results.boxes.conf)

                # Display FPS
                fps = 1.0 / (results.speed['inference'] / 1000.0)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Show the frame
                cv2.imshow('YOLOv8 Real-Time Detection', frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error during inference: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = '/home/phatnguyen/Documents/YOLO/HS001/runs/detect/train/weights/best.pt'
    confidence = 0.9
    epsilon = 0.1
    resize_height = 640
    resize_width = 640
    yolo_realtime = RealTimeYOLO(
        model_path=model_path,
        confidence=confidence,
        epsilon=epsilon,
        resize_height=resize_height,
        resize_width=resize_width
    )

    yolo_realtime.run()
    