# Underwater Object Detection with YOLO

This project is a **Streamlit-based web app** for detecting underwater objects using a YOLO model trained on custom data.  
You can upload **images or videos**, and the app will display both the original and annotated outputs.

![Streamlit Demo](Demo.gif)

---

## Features
- Upload **images (.jpg, .jpeg, .png)** or **videos (.mp4, .avi, .mov)**.
- Detect underwater objects using a pre-trained YOLO model (`best.pt`).
- Shows **original vs detected results**.
- Automatically clears old prediction folders for clean outputs.

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SnehasishKabi/UnderwaterObjectDetection.git
    cd UnderwaterObjectDetection
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # source venv/bin/activate   # On Mac/Linux
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Make sure your YOLO model weights (`best.pt`) are in the project folder.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. Open your browser at `http://localhost:8501`.
4. Upload an image and view the detection results.

---


