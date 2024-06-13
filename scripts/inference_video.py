import torch
import torchvision.transforms as transforms
import cv2
from model import create_ssd_model
import config
from dataset import VOC_CLASSES_REVERSE
import random

# Define a color map for each class
COLOR_MAP = {
    1: (0, 0, 255),     # aeroplane - red
    2: (0, 255, 0),     # bicycle - green
    3: (255, 0, 0),     # bird - blue
    4: (0, 255, 255),   # boat - yellow
    5: (255, 0, 255),   # bottle - magenta
    6: (255, 255, 0),   # bus - cyan
    7: (128, 0, 0),     # car - maroon
    8: (0, 128, 0),     # cat - dark green
    9: (0, 0, 128),     # chair - navy
    10: (128, 128, 0),  # cow - olive
    11: (128, 0, 128),  # diningtable - purple
    12: (0, 128, 128),  # dog - teal
    13: (192, 192, 192),# horse - silver
    14: (128, 128, 128),# motorbike - gray
    15: (255, 165, 0),  # person - orange
    16: (0, 255, 127),  # pottedplant - spring green
    17: (216, 191, 216),# sheep - thistle
    18: (255, 20, 147), # sofa - deep pink
    19: (123, 104, 238),# train - medium slate blue
    20: (0, 191, 255)   # tvmonitor - deep sky blue
}


# Function to load the model
def load_model(model_path, num_classes):
    model = create_ssd_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

# Function to perform inference on a frame
def run_inference(model, frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions

def visualize_predictions(frame, predictions):
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Only visualize predictions with high confidence
            x1, y1, x2, y2 = box.int().numpy()
            class_name = VOC_CLASSES_REVERSE[label.item()]  # Map label to class name
            color = COLOR_MAP[label.item()]  # Get the color for the class
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# Main function to process a video
def process_video(model_path, video_path, output_path):
    # Load the model
    model = load_model(model_path, config.num_classes)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on the frame
        predictions = run_inference(model, frame)
        
        # Visualize predictions
        frame = visualize_predictions(frame, predictions)
        
        # Write the frame to the output video
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

# Run the main function
if __name__ == "__main__":
    model_path = 'ssd_model_300.pth'  # Path to model
    video_path = 'kikko_park.mp4'  # Path to the input video
    output_path = 'kikko_output.mp4'  # Path to save the output video 

    process_video(model_path, video_path, output_path)

