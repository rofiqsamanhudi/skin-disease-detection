import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet101, convnext_base
from timm import create_model
from PIL import Image
import pickle
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import io

# ============================
# PAGE CONFIG HARUS DI ATAS SEGALA SESUATU!
# ============================
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# DEVICE & PATH MODEL ABSOLUT
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "EfficientNet Custom": r"D:\Skin Disease Detection\model\cnn_skin_disease.pkl",
    "ConvNeXt-Base": r"D:\Skin Disease Detection\model\convnext_base_skin_disease_best.pkl",
    "ResNet-101": r"D:\Skin Disease Detection\model\resnet101_skin_disease_best.pkl",
}

CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis',
    'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer',
    'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors',
    'Vasculitis', 'Vitiligo', 'Warts'
]

MODEL_LEADERBOARD = [
    {"Model": "ResNet-101", "Val Accuracy": 78.96},
    {"Model": "ConvNeXt-Base", "Val Accuracy": 75.00},
    {"Model": "EfficientNet Custom", "Val Accuracy": 77.50},
]

# ============================
# DARK THEME CSS
# ============================
st.markdown('''
<style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stApp {background: linear-gradient(to bottom, #0E1117, #1A2332);}
    h1, h2, h3 {color: #00D4B8; font-weight: 600;}
    section[data-testid="stSidebar"] {background-color: #1A2332; border-right: 1px solid #262730;}
    .stButton > button {
        background-color: #00D4B8; color: #0E1117; border-radius: 12px;
        font-weight: 600; box-shadow: 0 4px 12px rgba(0,212,184,0.3);
        width: 100%; padding: 0.7rem;
    }
    .card {background: #1A2332; padding: 1.8rem; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.4); border: 1px solid #262730; margin-bottom: 1.5rem;}
</style>
''', unsafe_allow_html=True)

# ============================
# PREPROCESSING
# ============================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================
# CPU-SAFE UNPICKLER
# ============================
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# ============================
# LOAD MODEL (WITH STRICT=FALSE FOR COMPATIBILITY)
# ============================
@st.cache_resource
def load_model(model_path, model_name):
    with open(model_path, 'rb') as f:
        if torch.cuda.is_available():
            data = pickle.load(f)
        else:
            data = CPUUnpickler(f).load()

    if isinstance(data, dict):
        state_dict = data.get('model_state_dict') or data.get('state_dict') or data
    else:
        model = data
        model.to(DEVICE)
        model.eval()
        return model

    # Architecture
    if "resnet101" in model_name.lower():
        model = resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 22)
    elif "convnext" in model_name.lower():
        model = convnext_base(pretrained=False)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 22)
    else:
        # EfficientNet-B0 dari timm
        model = create_model('efficientnet_b0', pretrained=False, num_classes=22)

    # Load dengan strict=False agar ignore mismatch
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# Load semua model
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        with st.spinner(f"Memuat {name}..."):
            models[name] = load_model(path, name)
    else:
        st.error(f"File tidak ditemukan: {path}")
        st.stop()

# ============================
# GRAD-CAM & OVERLAY
# ============================
def generate_heatmap(model, img_tensor):
    try:
        img_tensor = img_tensor.clone().detach().requires_grad_(True)
        activations = []
        gradients = []

        def forward_hook(m, i, o):
            activations.append(o.detach())
        def backward_hook(m, gi, go):
            if go[0] is not None:
                gradients.append(go[0].detach())

        target_layer = None
        for name, module in model.named_modules():
            if "layer4" in name or "stages" in name or "features" in name or "blocks" in name:
                target_layer = module
                break
        if target_layer is None:
            return None

        fwd = target_layer.register_forward_hook(forward_hook)
        bwd = target_layer.register_backward_hook(backward_hook)

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
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    except:
        return None
    finally:
        try:
            fwd.remove()
            bwd.remove()
        except:
            pass

def overlay_heatmap(img_np, cam):
    if cam is None:
        return img_np
    
    # Resize cam ke ukuran gambar asli (height, width)
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = heatmap * 0.4 + img_np * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ============================
# SIDEBAR & MAIN UI
# ============================
with st.sidebar:
    st.title("🧴 Skin Disease Detection")
    uploaded = st.file_uploader("Upload gambar lesi kulit", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Pilih Model", options=list(models.keys()))
    analyze_btn = st.button("🔍 Analyze Image", disabled=uploaded is None, use_container_width=True)
    st.markdown("---")
    st.caption(f"Device: {DEVICE}")
    st.warning("⚠️ Hanya untuk edukasi. Konsultasi dokter kulit wajib.")
    st.markdown("---")
    st.subheader("🏆 Leaderboard")
    df_lb = pd.DataFrame(MODEL_LEADERBOARD)
    st.dataframe(df_lb.style.format({"Val Accuracy": "{:.2f}%"}), use_container_width=True, hide_index=True)

st.title("Skin Disease Classification Dashboard")
st.markdown("### Deteksi 22 Jenis Penyakit Kulit dengan Deep Learning")

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Gambar Input", use_container_width=True)
    with col2:
        heatmap_placeholder = st.empty()

    if analyze_btn or 'result' in st.session_state:
        if analyze_btn:
            st.session_state.clear()
            with st.spinner(f"Menganalisis dengan {method}..."):
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
            if result['overlaid'] is not None:
                heatmap_placeholder.image(result['overlaid'], caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                heatmap_placeholder.info("Heatmap tidak tersedia")

        prob_df = pd.DataFrame({
            "Penyakit": CLASS_NAMES,
            "Probabilitas": result['probs']
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

        cola, colb = st.columns([1, 2])
        with cola:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔬 Prediksi")
            st.metric("Penyakit", result['pred_label'])
            st.metric("Confidence", f"{max(result['probs'])*100:.2f}%")
            st.subheader("Top 5")
            for i in range(5):
                row = prob_df.iloc[i]
                st.metric(row["Penyakit"], f"{row['Probabilitas']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with colb:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Probabilitas")
            chart = st.radio("Chart", ["Bar", "Pie"], horizontal=True)
            top10 = prob_df.head(10)
            if chart == "Bar":
                fig = px.bar(top10[::-1], x="Probabilitas", y="Penyakit", orientation="h")
            else:
                fig = px.pie(top10, values="Probabilitas", names="Penyakit", hole=0.4)
            fig.update_layout(height=500, plot_bgcolor="#1A2332", paper_bgcolor="#1A2332", font_color="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload gambar untuk memulai")

st.markdown("---")
st.caption("Dibuat dengan Streamlit | Model: ResNet-101, ConvNeXt-Base, EfficientNet")