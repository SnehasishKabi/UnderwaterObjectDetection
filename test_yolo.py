from ultralytics import YOLO
import cv2
import os
import glob

# Load your trained YOLO model
model = YOLO("best.pt")  # Make sure best.pt is in the same folder

# Path to your single test image
image_path = r"C:\Users\sneha\OneDrive\Desktop\deep_learning\test\images\1bc7-iudfmpmn7245599_jpg.rf.0c27f6617b1c2d7665a4badbb8474e28.jpg"

# Run prediction
results = model.predict(source=image_path, conf=0.25, save=True)

# Find the latest prediction folder
latest_folder = max(glob.glob('runs/detect/predict*'), key=os.path.getmtime)
predicted_image = os.path.join(latest_folder, os.path.basename(image_path))

# Display the predicted image
img = cv2.imread(predicted_image)
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Prediction saved at: {predicted_image}")
