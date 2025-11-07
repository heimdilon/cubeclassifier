import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import sys

# Check if model file exists
MODEL_PATH = "cube_classifier_rpi.pt"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please ensure you have transferred the model file to this directory.")
    sys.exit(1)

# Load the TorchScript model
try:
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define preprocessing transforms for grayscale images
transform = transforms.Compose([
    transforms.Resize((240, 320)),  # Resize to 240x320
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
])

# Class names
class_names = ['good', 'defective']

def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to PIL Image
    pil_image = Image.fromarray(gray_image)
    # Apply transforms
    tensor_image = transform(pil_image)
    # Add batch dimension
    batch_image = tensor_image.unsqueeze(0)
    return batch_image

def predict_cube(image):
    """Predict if cube is good or defective"""
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return class_names[predicted_class], confidence

def main():
    # Initialize camera (you might need to adjust the camera index)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to 320x240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting cube detection. Press 'q' to quit.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Make prediction
        start_time = time.time()
        prediction, confidence = predict_cube(frame)
        inference_time = time.time() - start_time
        
        # Display result on frame
        label = f"{prediction}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {inference_time*1000:.1f}ms", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Cube Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()