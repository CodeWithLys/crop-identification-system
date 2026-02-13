import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import sys
from pathlib import Path

# Add YOLOv5 to path
yolo_path = Path("Final Trained Model/yolov5-master")
sys.path.insert(0, str(yolo_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Page config
st.set_page_config(
    page_title="Crop Identification Tool - Alyssa Krishna",
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
        font-weight: bold;
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        device = select_device('cpu')  # Force CPU for Streamlit Cloud
        weights_path = "Final Trained Model/yolov5-master/runs/train/exp6/weights/best.pt"
        data_path = "Final Trained Model/yolov5-master/dataset.yaml"
        
        model = DetectMultiBackend(
            weights=weights_path,
            device=device,
            data=data_path
        )
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Header
st.markdown('<h1 class="main-header">🌾 Crop Identification Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered crop detection using YOLOv5 unsupervised learning</p>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png", width=200)
    st.markdown("### About")
    st.info("""
    Built by **Alyssa Krishna**
    
    This tool uses a custom-trained YOLOv5 model for crop identification using unsupervised learning techniques.
    
    **Tech Stack:**
    - YOLOv5
    - PyTorch
    - Streamlit
    - OpenCV
    """)
    
    st.markdown("### Supported Crops")
    crops = ['🍎 Apple', '🍌 Banana', '🥕 Carrot', '🌽 Corn', '🍇 Grapes', 
             '🥝 Kiwi', '🥬 Lettuce', '🧅 Onion', '🍍 Pineapple', '🥔 Potato', '🍅 Tomato']
    for crop in crops:
        st.write(crop)

# Main content
st.markdown("### 📤 Upload Crop Image")
uploaded_file = st.file_uploader("Choose an image containing crops", type=["jpg", "jpeg", "png"], help="Upload a clear image of one or more crops")

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Run detection
    with st.spinner('🔍 Analyzing image with AI...'):
        try:
            # Load model
            model, device = load_model()
            
            if model is None:
                st.error("Failed to load model. Please check the model path.")
                st.stop()
            
            # Prepare image
            img0 = np.array(image)
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            
            # Letterbox
            img = letterbox(img0, 640, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            
            # Convert to tensor
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            pred = model(img, augment=False, visualize=False)
            
            # NMS
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
            
            # Process detections
            detections = []
            annotator = Annotator(img0, line_width=3, example=str(model.names))
            
            det = pred[0]
            
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    detections.append({
                        'crop': model.names[c],
                        'confidence': conf.item(),
                        'bbox': [int(x) for x in xyxy]
                    })
            
            # Show annotated results
            with col2:
                st.markdown("#### Detection Results")
                result_img = annotator.result()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)
            
            # Show detection details
            st.markdown("---")
            if detections:
                st.success(f"✅ Successfully detected {len(detections)} crop(s)!")
                
                cols = st.columns(min(len(detections), 3))
                
                for i, det in enumerate(detections):
                    col_idx = i % 3
                    with cols[col_idx]:
                        emoji_map = {
                            'apple': '🍎', 'banana': '🍌', 'carrot': '🥕', 
                            'corn': '🌽', 'grapes': '🍇', 'kiwi': '🥝',
                            'lettuce': '🥬', 'onion': '🧅', 'pineapple': '🍍',
                            'potato': '🥔', 'tomato': '🍅'
                        }
                        emoji = emoji_map.get(det['crop'].lower(), '🌾')
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
                            <h3 style="margin: 10px 0;">{det['crop'].capitalize()}</h3>
                            <p style="font-size: 1.2rem; margin: 5px 0;">
                                Confidence: <strong>{det['confidence']:.1%}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No crops detected in the image. Try uploading a clearer image with visible crops.")
                
        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")
            st.info("💡 Tip: Make sure your image contains one or more of the supported crops.")

else:
    # Show example/instructions when no file uploaded
    st.info("👆 Upload an image above to start detecting crops!")
    
    st.markdown("### 💡 Tips for Best Results")
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **✅ Good Images:**
        - Clear, well-lit photos
        - Crops clearly visible
        - Single or multiple crops
        - Close-up shots work best
        """)
    
    with tips_col2:
        st.markdown("""
        **❌ Avoid:**
        - Blurry images
        - Very dark/overexposed photos
        - Extreme angles
        - Heavily processed images
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Crop Identification Tool</strong> | Built with YOLOv5 & Streamlit</p>
    <p>Developed by <strong>Alyssa Krishna</strong> | 
    <a href="https://github.com/CodeWithLys/crop-identification-system" target="_blank">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
