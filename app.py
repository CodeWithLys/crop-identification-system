import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys

# Add YOLOv5 to path
sys.path.append('./yolov5-master')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Page config
st.set_page_config(
    page_title="Crop Identification Tool",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    device = select_device('')
    model = DetectMultiBackend(
        weights='runs/train/exp6/weights/best.pt',
        device=device,
        data='dataset.yaml'
    )
    return model, device

# Header
st.markdown('<h1 class="main-header">🌾 Crop Identification Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image of a crop and let AI identify it using YOLOv5</p>', unsafe_allow_html=True)

# Info section
with st.expander("ℹ️ About This Tool"):
    st.write("""
    This tool uses a custom-trained YOLOv5 model to identify crops in images.
    
    **Supported Crops:**
    - 🍎 Apple
    - 🍌 Banana  
    - 🥕 Carrot
    - 🌽 Corn
    - 🍇 Grapes
    - 🥝 Kiwi
    - 🥬 Lettuce
    - 🧅 Onion
    - 🍍 Pineapple
    - 🥔 Potato
    - 🍅 Tomato
    
    **How it works:**
    1. Upload an image containing one or more crops
    2. The model detects and identifies each crop
    3. View the results with bounding boxes and confidence scores
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Run detection
    with st.spinner('🔍 Analyzing crops...'):
        try:
            # Load model
            model, device = load_model()
            
            # Prepare image
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Inference
            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.permute(2, 0, 1).float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # Predict
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            
            # Process detections
            detections = []
            annotator = Annotator(img, line_width=3)
            
            if len(pred[0]):
                # Rescale boxes
                pred[0][:, :4] = scale_boxes(img_tensor.shape[2:], pred[0][:, :4], img.shape).round()
                
                # Draw boxes and collect results
                for *xyxy, conf, cls in reversed(pred[0]):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    detections.append({
                        'crop': model.names[c],
                        'confidence': f'{conf:.2%}'
                    })
            
            # Show results
            with col2:
                st.subheader("Detection Results")
                result_img = annotator.result()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)
            
            # Show detection details
            if detections:
                st.success(f"✅ Found {len(detections)} crop(s)!")
                
                for i, det in enumerate(detections, 1):
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Detection {i}</h3>
                        <p style="font-size: 1.5rem; margin: 10px 0;">
                            <strong>{det['crop'].capitalize()}</strong>
                        </p>
                        <p style="font-size: 1.2rem;">
                            Confidence: {det['confidence']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No crops detected. Try another image!")
                
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            st.info("Make sure the model weights file is in the correct location: runs/train/exp6/weights/best.pt")

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666;">
    Built with YOLOv5 | Developed by Alyssa Krishna
</p>
""", unsafe_allow_html=True)
```

---

## Step 3: Create requirements.txt

In the same repo, create `requirements.txt`:
```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
Pillow==10.0.0
numpy==1.24.3
PyYAML==6.0.1
