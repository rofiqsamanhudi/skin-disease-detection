import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet101
from timm import create_model
from PIL import Image
import pickle
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import os
import io

# ============================
# PAGE CONFIG & TACTICAL CSS
# ============================
st.set_page_config(page_title="Skin Disease Detection", layout="wide", initial_sidebar_state="expanded")

def inject_tactical_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');
        
        .block-container {
            padding-top: 4rem !important;
            padding-bottom: 3rem !important;
        }
        
        .stApp {
            background-color: #0b0d11;
            color: #aeb9cc;
            font-family: 'Roboto Mono', monospace;
            font-size: 11px;
        }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0b0d11; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
        
        h1 { font-size: 28px !important; color: #e6edf3; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; margin: 0; }
        h2 { font-size: 18px !important; color: #58a6ff; letter-spacing: 1px; text-transform: uppercase; margin-top: 20px; }
        h3 { font-size: 14px !important; color: #8b949e; text-transform: uppercase; font-weight: 600; }
        
        section[data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid #30363d;
        }
        
        .stButton > button {
            background-color: #1f6feb;
            color: white;
            border: none;
            border-radius: 0px;
            padding: 10px 20px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: bold;
            width: 100%;
        }
        .stButton > button:hover { background-color: #388bfd; }
        
        .card {
            background: #161b22;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .close-btn {
            position: absolute;
            top: 8px;
            right: 12px;
            background: none !important;
            border: none !important;
            font-size: 20px !important;
            color: #8b949e !important;
            cursor: pointer;
            padding: 0 !important;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s;
        }
        .close-btn:hover {
            color: #f85149 !important;
            background-color: rgba(248, 81, 73, 0.1) !important;
        }
        
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #010409;
            color: #484f58;
            text-align: right;
            padding: 8px 20px;
            font-size: 9px;
            border-top: 1px solid #30363d;
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)

inject_tactical_styles()

# ============================
# HEADER
# ============================
st.markdown("<h1>Skin Disease Classification System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ============================
# DEVICE & PATHS
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ResNet-101": r"D:\Skin Disease Detection\model\resnet101_skin_disease.pkl",
    "ConvNeXt Base": r"D:\Skin Disease Detection\model\convnext_base_skin_disease.pkl",
    "CNN Scratch": r"D:\Skin Disease Detection\model\cnn_skin_disease.pkl",
}

ASSET_PATHS = {
    "ResNet-101": {
        "curve": r"D:\Skin Disease Detection\assets\resnet_curve.png",
        "matrix": r"D:\Skin Disease Detection\assets\resnet_matrix.png"
    },
    "ConvNeXt Base": {
        "curve": r"D:\Skin Disease Detection\assets\convnext_curve.png",
        "matrix": r"D:\Skin Disease Detection\assets\convnext_matrix.png"
    },
    "CNN Scratch": {
        "curve": r"D:\Skin Disease Detection\assets\cnn_curve.png",
        "matrix": r"D:\Skin Disease Detection\assets\cnn_matrix.png"
    }
}

CLASS_NAMES = [
    'Acne', 'Actinic Keratosis', 'Benign Tumors', 'Bullous Disease', 'Candidiasis',
    'Drug Eruption', 'Eczema', 'Infestations & Bites', 'Lichen Planus', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrheic Keratoses', 'Skin Cancer',
    'Sun Damage', 'Tinea', 'Normal Skin', 'Vascular Tumors',
    'Vasculitis', 'Vitiligo', 'Warts'
]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# ============================
# CNN SCRATCH ARCHITECTURE
# ============================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=22):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================
# HAPUS PREFIX
# ============================
def remove_backbone_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
            new_state_dict[new_key] = v
    return new_state_dict

# ============================
# LOAD MODELS
# ============================
models = {}
missing = []

def load_generic_model(path, create_fn):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f) if torch.cuda.is_available() else CPUUnpickler(f).load()
        full_sd = data['model_state_dict']
        cleaned_sd = remove_backbone_prefix(full_sd)
        model = create_fn()
        model.load_state_dict(cleaned_sd, strict=False)
        model.to(DEVICE)
        return model
    except Exception as e:
        raise e

try:
    model = load_generic_model(MODEL_PATHS["ResNet-101"], lambda: resnet101(pretrained=False))
    model.fc = nn.Linear(model.fc.in_features, 22)
    model.to(DEVICE)
    models["ResNet-101"] = model
except Exception as e:
    missing.append(f"ResNet-101 → {str(e)}")

try:
    model = load_generic_model(MODEL_PATHS["ConvNeXt Base"], lambda: create_model('convnext_base', pretrained=False, num_classes=1000))
    model.head.fc = nn.Linear(model.head.fc.in_features, 22)
    model.to(DEVICE)
    models["ConvNeXt Base"] = model
except Exception as e:
    missing.append(f"ConvNeXt Base → {str(e)}")

try:
    model = load_generic_model(MODEL_PATHS["CNN Scratch"], lambda: CustomCNN(num_classes=22))
    models["CNN Scratch"] = model
except Exception as e:
    missing.append(f"CNN Scratch → {str(e)}")

if not models:
    st.error("Tidak ada model yang berhasil dimuat.")
    st.stop()

available_models = list(models.keys())

# ============================
# FULL CLASSIFICATION REPORTS
# ============================
FULL_CLASSIFICATION_REPORTS = {
    "CNN Scratch": """
**=== Classification Report - CNN Scratch ===**

| Class                  | Precision | Recall  | F1-score | Support |
|------------------------|-----------|---------|----------|---------|
| Acne                   | 0.5000    | 0.6615  | 0.5695   | 65      |
| Actinic Keratosis      | 0.4286    | 0.5422  | 0.4787   | 83      |
| Benign Tumors          | 0.3246    | 0.6116  | 0.4241   | 121     |
| Bullous Disease        | 0.3256    | 0.2545  | 0.2857   | 55      |
| Candidiasis            | 0.5000    | 0.3704  | 0.4255   | 27      |
| Drug Eruption          | 0.5106    | 0.3934  | 0.4444   | 61      |
| Eczema                 | 0.4015    | 0.4911  | 0.4418   | 112     |
| Infestations & Bites   | 0.3659    | 0.2500  | 0.2970   | 60      |
| Lichen Planus          | 0.4722    | 0.2787  | 0.3505   | 61      |
| Lupus                  | 0.3846    | 0.1471  | 0.2128   | 34      |
| Moles                  | 0.7857    | 0.2750  | 0.4074   | 40      |
| Psoriasis              | 0.4312    | 0.5341  | 0.4772   | 88      |
| Rosacea                | 0.5625    | 0.6429  | 0.6000   | 28      |
| Seborrheic Keratoses   | 0.5000    | 0.4118  | 0.4516   | 51      |
| Skin Cancer            | 0.4028    | 0.3766  | 0.3893   | 77      |
| Sun Damage             | 0.3333    | 0.1176  | 0.1739   | 34      |
| Tinea                  | 0.4299    | 0.4510  | 0.4402   | 102     |
| Normal Skin            | 0.9261    | 0.8624  | 0.8932   | 189     |
| Vascular Tumors        | 0.2642    | 0.2333  | 0.2478   | 60      |
| Vasculitis             | 0.5000    | 0.3654  | 0.4222   | 52      |
| Vitiligo               | 0.7922    | 0.7439  | 0.7673   | 82      |
| Warts                  | 0.5000    | 0.4531  | 0.4754   | 64      |

**Average Metrics**  
| Metric     | Macro Avg | Weighted Avg |
|------------|-----------|--------------|
| Precision  | 0.4837    | 0.5106       |
| Recall     | 0.4303    | 0.4942       |
| F1-score   | 0.4398    | 0.4891       |

**Overall Accuracy: 49.42%**
    """,
    "ConvNeXt Base": """
**=== Classification Report - ConvNeXt Base ===**

| Class                  | Precision | Recall  | F1-score | Support |
|------------------------|-----------|---------|----------|---------|
| Acne                   | 0.8696    | 0.9231  | 0.8955   | 65      |
| Actinic Keratosis      | 0.8133    | 0.7349  | 0.7722   | 83      |
| Benign Tumors          | 0.7368    | 0.8099  | 0.7717   | 121     |
| Bullous Disease        | 0.7407    | 0.7273  | 0.7339   | 55      |
| Candidiasis            | 0.6923    | 0.6667  | 0.6792   | 27      |
| Drug Eruption          | 0.7119    | 0.6885  | 0.7000   | 61      |
| Eczema                 | 0.7177    | 0.7946  | 0.7542   | 112     |
| Infestations & Bites   | 0.6308    | 0.6833  | 0.6560   | 60      |
| Lichen Planus          | 0.6613    | 0.6721  | 0.6667   | 61      |
| Lupus                  | 0.8125    | 0.3824  | 0.5200   | 34      |
| Moles                  | 0.6829    | 0.7000  | 0.6914   | 40      |
| Psoriasis              | 0.8333    | 0.7386  | 0.7831   | 88      |
| Rosacea                | 0.5238    | 0.7857  | 0.6286   | 28      |
| Seborrheic Keratoses   | 0.8889    | 0.7843  | 0.8333   | 51      |
| Skin Cancer            | 0.7463    | 0.6494  | 0.6944   | 77      |
| Sun Damage             | 0.6216    | 0.6765  | 0.6479   | 34      |
| Tinea                  | 0.7551    | 0.7255  | 0.7400   | 102     |
| Normal Skin            | 0.9637    | 0.9841  | 0.9738   | 189     |
| Vascular Tumors        | 0.7119    | 0.7000  | 0.7059   | 60      |
| Vasculitis             | 0.6607    | 0.7115  | 0.6852   | 52      |
| Vitiligo               | 0.9620    | 0.9268  | 0.9441   | 82      |
| Warts                  | 0.7500    | 0.7969  | 0.7727   | 64      |

**Average Metrics**  
| Metric     | Macro Avg | Weighted Avg |
|------------|-----------|--------------|
| Precision  | 0.7494    | 0.7792       |
| Recall     | 0.7392    | 0.7743       |
| F1-score   | 0.7386    | 0.7735       |

**Overall Test Accuracy: 77.43%**
    """,
    "ResNet-101": """
**=== Classification Report - ResNet-101 ===**

| Class                  | Precision | Recall  | F1-score | Support |
|------------------------|-----------|---------|----------|---------|
| Acne                   | 0.8630    | 0.9692  | 0.9130   | 65      |
| Actinic Keratosis      | 0.7791    | 0.8072  | 0.7929   | 83      |
| Benign Tumors          | 0.6889    | 0.7686  | 0.7266   | 121     |
| Bullous Disease        | 0.7708    | 0.6727  | 0.7184   | 55      |
| Candidiasis            | 0.6364    | 0.7778  | 0.7000   | 27      |
| Drug Eruption          | 0.7797    | 0.7541  | 0.7667   | 61      |
| Eczema                 | 0.7652    | 0.7857  | 0.7753   | 112     |
| Infestations & Bites   | 0.7200    | 0.6000  | 0.6545   | 60      |
| Lichen Planus          | 0.7600    | 0.6230  | 0.6847   | 61      |
| Lupus                  | 0.8182    | 0.5294  | 0.6429   | 34      |
| Moles                  | 0.6923    | 0.6750  | 0.6835   | 40      |
| Psoriasis              | 0.8161    | 0.8068  | 0.8114   | 88      |
| Rosacea                | 0.7742    | 0.8571  | 0.8136   | 28      |
| Seborrheic Keratoses   | 0.8750    | 0.8235  | 0.8485   | 51      |
| Skin Cancer            | 0.7534    | 0.7143  | 0.7333   | 77      |
| Sun Damage             | 0.6667    | 0.7059  | 0.6857   | 34      |
| Tinea                  | 0.7547    | 0.7843  | 0.7692   | 102     |
| Normal Skin            | 0.9738    | 0.9841  | 0.9789   | 189     |
| Vascular Tumors        | 0.7119    | 0.7000  | 0.7059   | 60      |
| Vasculitis             | 0.5938    | 0.7308  | 0.6552   | 52      |
| Vitiligo               | 0.9630    | 0.9512  | 0.9571   | 82      |
| Warts                  | 0.8333    | 0.7812  | 0.8065   | 64      |

**Average Metrics**  
| Metric     | Macro Avg | Weighted Avg |
|------------|-----------|--------------|
| Precision  | 0.7722    | 0.7941       |
| Recall     | 0.7637    | 0.7917       |
| F1-score   | 0.7647    | 0.7908       |

**Overall Test Accuracy: 79.17%**
    """
}

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("### SYSTEM CONTROL")
    uploaded = st.file_uploader("UPLOAD SKIN LESION IMAGE", type=["jpg", "jpeg", "png"])
    method = st.selectbox("SELECT ACTIVE MODEL", options=available_models)
    analyze_btn = st.button("EXECUTE ANALYSIS", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### MODEL LEADERBOARD")
    leaderboard_data = [
        {"Model": "ResNet-101", "Test Accuracy": "79.17%", "Training Info": "Total epochs trained: 45 | Final Learning Rate: 1e-06"},
        {"Model": "ConvNeXt Base", "Test Accuracy": "77.43%", "Training Info": "Total epochs: 30 | Final Learning Rate: 1e-04"},
        {"Model": "CNN Scratch", "Test Accuracy": "49.42%", "Training Info": "Total epochs trained: 200 | Final Learning Rate: 6e-05"},
    ]
    df_lb = pd.DataFrame(leaderboard_data)
    st.dataframe(df_lb, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    if st.button("CLASSIFICATION REPORT", use_container_width=True):
        st.session_state.show_class_report = not st.session_state.get("show_class_report", False)
        st.session_state.show_dashboard = False
    
    if st.button("DASHBOARD REPORT", use_container_width=True):
        st.session_state.show_dashboard = not st.session_state.get("show_dashboard", False)
        st.session_state.show_class_report = False
    
    st.markdown("---")
    st.caption(f"DEVICE: {DEVICE}")
    st.caption("FOR EDUCATIONAL PURPOSES ONLY")

# ============================
# MAIN TITLE
# ============================
st.markdown("<h2 style='text-align: center;'>Deep Learning-Based Skin Disease Classification</h2>", unsafe_allow_html=True)

# ============================
# CLASSIFICATION REPORT
# ============================
if st.session_state.get("show_class_report", False):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_title, col_close = st.columns([10, 1])
    with col_title:
        st.markdown("<h2 style='color: #58a6ff; margin:0;'>CLASSIFICATION REPORTS</h2>", unsafe_allow_html=True)
    with col_close:
        if st.button("×", key="close_class"):
            st.session_state.show_class_report = False
            st.rerun()
    
    st.markdown("---")
    for model_name, report in FULL_CLASSIFICATION_REPORTS.items():
        with st.expander(f"📊 {model_name}", expanded=True):
            st.markdown(report)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# DASHBOARD REPORT
# ============================
if st.session_state.get("show_dashboard", False):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_title, col_close = st.columns([10, 1])
    with col_title:
        st.markdown("<h2 style='color: #58a6ff; margin:0;'>MODEL TRAINING & EVALUATION DASHBOARD</h2>", unsafe_allow_html=True)
    with col_close:
        if st.button("×", key="close_dash"):
            st.session_state.show_dashboard = False
            st.rerun()
    
    st.markdown("---")
    for model_name in ["ResNet-101", "ConvNeXt Base", "CNN Scratch"]:
        assets = ASSET_PATHS[model_name]
        st.markdown(f"### {model_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(assets["curve"]):
                st.image(assets["curve"], caption="Training & Validation Curves", use_container_width=True)
            else:
                st.warning("Curve image not found.")
        
        with col2:
            if os.path.exists(assets["matrix"]):
                st.image(assets["matrix"], caption="Confusion Matrix", use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")
        
        st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# EXECUTE ANALYSIS → TUTUP REPORT OTOMATIS
# ============================
if analyze_btn:
    # Tutup semua report saat analysis dimulai
    st.session_state.show_class_report = False
    st.session_state.show_dashboard = False

# ============================
# GRAD-CAM & PREDICTION
# ============================
def generate_heatmap(model, img_tensor):
    if "CNN Scratch" in str(type(model)):
        # Support for CNN Scratch - hook to the last Conv2d in features
        target_layer = None
        for module in reversed(model.features):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
        if target_layer is None:
            return None
    else:
        target_layer = None
        for name, module in model.named_modules():
            if "layer4" in name or "stages.3" in name:
                target_layer = module
                break
        
        if target_layer is None:
            return None
    
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    activations = []
    gradients = []

    def forward_hook(m, i, o):
        activations.append(o.detach())
    def backward_hook(m, gi, go):
        if go[0] is not None:
            gradients.append(go[0].detach())

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    try:
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, pred_class].backward()

        if len(gradients) == 0 or len(activations) == 0:
            return None

        grad = gradients[0].mean(dim=[1, 2], keepdim=True)
        act = activations[0]
        cam = (grad * act).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam
    finally:
        fwd.remove()
        bwd.remove()

def overlay_heatmap(img_np, cam):
    if cam is None:
        return img_np
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * 0.4 + img_np * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ============================
# MAIN CONTENT
# ============================
if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image_pil, caption="INPUT IMAGE", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn or 'result' in st.session_state:
        if analyze_btn:
            st.session_state.clear()

            with st.spinner(f"ANALYZING WITH {method.upper()}..."):
                model = models[method]
                x = preprocess(image_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                
                pred_idx = np.argmax(probs)
                pred_label = CLASS_NAMES[pred_idx]
                
                cam = generate_heatmap(model, x)
                overlaid = overlay_heatmap(image_np, cam)
                
                st.session_state['result'] = {
                    'probs': probs.tolist(),
                    'pred_label': pred_label,
                    'overlaid': overlaid,
                    'method': method
                }

        result = st.session_state['result']
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if result['overlaid'] is not None:
                st.image(result['overlaid'], caption=f"GRAD-CAM HEATMAP ({result['method'].upper()})", use_container_width=True)
                st.caption("RED/ORANGE = HIGH INFLUENCE REGION")
            else:
                st.info("GRAD-CAM ONLY AVAILABLE FOR RESNET-101 & CONVNEXT")
            st.markdown('</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Condition": CLASS_NAMES,
            "Probability": result['probs']
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("PREDICTION RESULT")
            st.metric("MOST LIKELY CONDITION", result['pred_label'])
            st.metric("CONFIDENCE LEVEL", f"{max(result['probs'])*100:.2f}%")
            st.subheader("TOP 5 PREDICTIONS")
            for i in range(5):
                row = prob_df.iloc[i]
                st.metric(row["Condition"], f"{row['Probability']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("PROBABILITY DISTRIBUTION")
            chart = st.radio("CHART TYPE", ["Bar Chart", "Pie Chart"], horizontal=True)
            top10 = prob_df.head(10)
            if chart == "Bar Chart":
                fig = px.bar(top10[::-1], x="Probability", y="Condition", orientation="h")
            else:
                fig = px.pie(top10, values="Probability", names="Condition", hole=0.4)
            fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c0c0")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.info("UPLOAD SKIN LESION IMAGE FROM SIDEBAR TO BEGIN ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# FOOTER
# ============================
st.markdown("""
<div class="footer">
    SYSTEM v1.0 // FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)