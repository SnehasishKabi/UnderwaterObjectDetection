import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess
import sys
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Underwater Object Detection",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "best.pt"
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
RUNS_DIR = "runs"
DETECT_DIR = os.path.join(RUNS_DIR, "detect")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['ultralytics', 'opencv-python', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'ultralytics':
                import ultralytics
            elif package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                from PIL import Image
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.info("Please install them using: pip install " + " ".join(missing_packages))
        return False
    return True

def check_model_exists():
    """Check if the YOLO model file exists"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in the current directory!")
        st.info("Please ensure your trained YOLO model 'best.pt' is in the same directory as this app.")
        return False
    return True

def clean_old_predictions():
    """Remove old prediction folders to prevent clutter"""
    try:
        if os.path.exists(RUNS_DIR):
            shutil.rmtree(RUNS_DIR)
            st.info("🧹 Cleaned up old prediction folders")
        return True
    except Exception as e:
        st.warning(f"Could not clean old predictions: {str(e)}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def run_yolo_detection(image_path):
    """Run YOLO detection on the uploaded image"""
    try:
        # Import YOLO here to avoid import errors if ultralytics is not installed
        from ultralytics import YOLO
        
        # Load the trained model
        model = YOLO(MODEL_PATH)
        
        # Run prediction
        with st.spinner("🔍 Running underwater object detection..."):
            results = model.predict(
                source=image_path,
                save=True,
                conf=0.25,  # Confidence threshold
                project=RUNS_DIR,
                name="detect",
                exist_ok=True
            )
        
        return results
    except Exception as e:
        st.error(f"Error during YOLO detection: {str(e)}")
        return None

def find_output_image():
    """Find the output image from YOLO detection"""
    try:
        # Look for the most recent detect folder
        detect_folders = []
        if os.path.exists(DETECT_DIR):
            detect_folders.append(DETECT_DIR)
        
        # Also check for numbered detect folders (detect2, detect3, etc.)
        for i in range(2, 10):  # Check up to detect9
            numbered_dir = f"{DETECT_DIR}{i}"
            if os.path.exists(numbered_dir):
                detect_folders.append(numbered_dir)
        
        if not detect_folders:
            return None
        
        # Get the most recently modified folder
        latest_folder = max(detect_folders, key=os.path.getmtime)
        
        # Find image files in the folder
        for file in os.listdir(latest_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(latest_folder, file)
        
        return None
    except Exception as e:
        st.error(f"Error finding output image: {str(e)}")
        return None

def load_image(image_path):
    """Load and return an image"""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🐠 Underwater Object Detection")
    st.markdown("Upload an underwater image to detect objects using your trained YOLO model")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Created by: Snehasish Kabi**
        
        This underwater object detection system helps marine researchers, conservationists, and diving enthusiasts identify and catalog marine life automatically. It's particularly useful for:
        
        - **Marine Biology Research**: Automated species identification in underwater footage
        - **Environmental Monitoring**: Tracking marine biodiversity and ecosystem health  
        - **Conservation Efforts**: Monitoring endangered species and habitat changes
        - **Educational Purposes**: Learning about marine life identification
        - **Diving Documentation**: Cataloging underwater discoveries
        
        **Features:**
        - Upload JPG, JPEG, or PNG images
        - Real-time object detection
        - Side-by-side comparison
        - Automatic cleanup of old predictions
        """)
        
        st.header("🔧 Model Info")
        if os.path.exists(MODEL_PATH):
            st.success(f"✅ Model loaded: {MODEL_PATH}")
            # Get file size
            model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
            st.info(f"Model size: {model_size:.1f} MB")
        else:
            st.error(f"❌ Model not found: {MODEL_PATH}")
    
    # Check requirements
    if not check_requirements() or not check_model_exists():
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an underwater image",
        type=SUPPORTED_FORMATS,
        help="Upload a JPG, JPEG, or PNG image for object detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image info
        st.success(f"📁 Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Save uploaded file
        temp_image_path = save_uploaded_file(uploaded_file)
        
        if temp_image_path:
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                original_image = load_image(temp_image_path)
                if original_image:
                    st.image(original_image, use_column_width=True)
            
            # Run detection button
            if st.button("🔍 Run Detection", type="primary"):
                # Clean old predictions
                clean_old_predictions()
                
                # Run YOLO detection
                results = run_yolo_detection(temp_image_path)
                
                if results:
                    # Find output image
                    output_image_path = find_output_image()
                    
                    if output_image_path:
                        with col2:
                            st.subheader("🎯 Detection Results")
                            detected_image = load_image(output_image_path)
                            if detected_image:
                                st.image(detected_image, use_column_width=True)
                        
                        # Display detection statistics
                        st.success("✅ Detection completed successfully!")
                        
                        # Show detection summary
                        if results and len(results) > 0:
                            result = results[0]  # Get first result
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                num_detections = len(result.boxes)
                                st.info(f"🎯 Found {num_detections} object(s) in the image")
                                
                                # Show confidence scores
                                if num_detections > 0:
                                    confidences = result.boxes.conf.cpu().numpy()
                                    avg_confidence = np.mean(confidences)
                                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                            else:
                                st.info("🔍 No objects detected in the image")
                    else:
                        st.error("Could not find detection output image")
                else:
                    st.error("Detection failed. Please check your model and try again.")
            
            # Clean up temporary file
            try:
                os.unlink(temp_image_path)
            except:
                pass  # Ignore cleanup errors
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an underwater image to get started")
        
        # Show example of what the app can detect
        st.subheader("🌊 What can this detect?")
        st.markdown("""
        Your trained YOLO model can detect various underwater objects such as:
        - Fish species
        - Marine life
        - Underwater debris
        - Coral formations
        - And more (depending on your training data)
        """)

if __name__ == "__main__":
    main()