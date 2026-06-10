from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input image
image_path = "input_image.jpg"

# Perform detection
results = model(image_path)

# Save output image
annotated_frame = results[0].plot()
cv2.imwrite("output_result.png", annotated_frame)

print("Detection completed. Output saved as output_result.png")
