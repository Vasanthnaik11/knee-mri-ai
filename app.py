
import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import joblib
import os

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Knee MRI AI Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 12px;
}

.positive {
    font-weight: 700;
}

.negative {
    font-weight: 700;
}

.metric-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    text-align: center;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">🦴 Knee MRI AI Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multimodal Knee Abnormality Detection using MRI + Radiology Report'
    '</div>',
    unsafe_allow_html=True
)

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This research prototype combines MRI image features "
        "with radiology report text to estimate probabilities "
        "for 12 knee abnormalities."
    )

    st.divider()

    st.subheader("Model")

    st.write("MRI encoder: ResNet18")
    st.write("MRI features: 512")
    st.write("Text features: TF-IDF")
    st.write("Vocabulary: 1000")
    st.write("Classifiers: 12")

    st.divider()

    st.subheader("Validation")

    st.metric(
        "Macro AUC",
        "0.7147"
    )

    st.divider()

    st.warning(
        "Research prototype only. "
        "This application is not a medical diagnostic system."
    )

# ================================================================
# DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ================================================================
# MODEL DIRECTORY
# ================================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "final_multimodal_models"
)

# ================================================================
# LOAD RESNET
# ================================================================

@st.cache_resource
def load_resnet():

    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(
        weights=weights
    )

    model.fc = torch.nn.Identity()

    model = model.to(device)
    model.eval()

    return model, weights.transforms()


resnet, transform = load_resnet()

# ================================================================
# LOAD MULTIMODAL MODELS
# ================================================================

@st.cache_resource
def load_models():

    tfidf = joblib.load(
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    text_selectors = joblib.load(
        os.path.join(
            MODEL_DIR,
            "text_selectors.pkl"
        )
    )

    logistic_models = joblib.load(
        os.path.join(
            MODEL_DIR,
            "logistic_models.pkl"
        )
    )

    label_columns = joblib.load(
        os.path.join(
            MODEL_DIR,
            "label_columns.pkl"
        )
    )

    return (
        tfidf,
        text_selectors,
        logistic_models,
        label_columns
    )


(
    tfidf,
    text_selectors,
    logistic_models,
    label_columns
) = load_models()

# ================================================================
# MRI FEATURE EXTRACTION
# ================================================================

def extract_mri_feature(volume):

    if volume.ndim != 3:

        raise ValueError(
            "MRI volume must have shape "
            "(slices, height, width)."
        )

    middle_slice = volume[
        volume.shape[0] // 2
    ]

    minimum = middle_slice.min()
    maximum = middle_slice.max()

    if maximum > minimum:

        normalized = (
            (middle_slice - minimum)
            / (maximum - minimum)
        )

    else:

        normalized = np.zeros_like(
            middle_slice
        )

    image = Image.fromarray(
        (normalized * 255).astype(np.uint8)
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        feature = resnet(
            tensor
        )

    return (
        feature
        .cpu()
        .numpy()
        .astype(np.float32)
    )

# ================================================================
# PREDICTION
# ================================================================

def predict_knee(volume, report):

    mri_feature = extract_mri_feature(
        volume
    )

    text_features = tfidf.transform(
        [str(report)]
    )

    predictions = {}

    for label in label_columns:

        selector = text_selectors[label]

        selected_text = selector.transform(
            text_features
        ).toarray()

        combined = np.concatenate(
            [
                mri_feature,
                selected_text
            ],
            axis=1
        )

        model = logistic_models[label]

        probability = model.predict_proba(
            combined
        )[0, 1]

        predictions[label] = float(
            probability
        )

    return predictions

# ================================================================
# INPUT SECTION
# ================================================================

st.header("📥 Input Data")

col1, col2 = st.columns(
    [1, 1]
)

with col1:

    st.subheader("🧲 MRI Volume")

    uploaded_file = st.file_uploader(
        "Upload processed MRI volume (.npy)",
        type=["npy"]
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

with col2:

    st.subheader("📝 Radiology Report")

    report = st.text_area(
        "Paste the radiology report",
        height=220,
        placeholder=(
            "Enter or paste the radiology report here..."
        )
    )

# ================================================================
# ANALYZE BUTTON
# ================================================================

st.divider()

analyze = st.button(
    "🔍 ANALYZE KNEE MRI",
    type="primary",
    use_container_width=True
)

# ================================================================
# ANALYSIS
# ================================================================

if analyze:

    if uploaded_file is None:

        st.error(
            "Please upload an MRI .npy volume."
        )

        st.stop()

    if not report.strip():

        st.error(
            "Please enter the radiology report."
        )

        st.stop()

    try:

        with st.spinner(
            "Analyzing MRI and radiology report..."
        ):

            volume = np.load(
                uploaded_file
            )

            predictions = predict_knee(
                volume,
                report
            )

        st.success(
            "✅ Analysis completed successfully."
        )

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        positive = {
            k: v
            for k, v in predictions.items()
            if v >= 0.50
        }

        st.header("📊 AI Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Abnormalities Analyzed",
                len(predictions)
            )

        with col2:

            st.metric(
                "Positive Predictions",
                len(positive)
            )

        with col3:

            st.metric(
                "Model Macro AUC",
                "0.7147"
            )

        # --------------------------------------------------------
        # POSITIVE FINDINGS
        # --------------------------------------------------------

        st.subheader(
            "🔴 Higher-Probability Findings"
        )

        if positive:

            for label, probability in sorted(
                positive.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                st.write(
                    f"**{label} — "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    probability
                )

        else:

            st.info(
                "No abnormalities exceeded the "
                "0.50 probability threshold."
            )

        # --------------------------------------------------------
        # ALL RESULTS
        # --------------------------------------------------------

        st.subheader(
            "📋 All Abnormality Predictions"
        )

        for label, probability in predictions.items():

            percentage = probability * 100

            if probability >= 0.50:

                status = "🔴 POSITIVE"

            else:

                status = "🟢 NEGATIVE"

            col1, col2, col3 = st.columns(
                [3, 2, 2]
            )

            with col1:

                st.write(
                    f"**{label}**"
                )

            with col2:

                st.write(
                    f"{percentage:.2f}%"
                )

            with col3:

                st.write(
                    status
                )

            st.progress(
                probability
            )

        # --------------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------------

        st.divider()

        st.warning(
            "⚠️ IMPORTANT: This is an AI research prototype. "
            "Predictions are probabilistic estimates and must "
            "not be used as a substitute for professional "
            "medical diagnosis or clinical decision-making."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )

# ================================================================
# FOOTER
# ================================================================

st.markdown(
    '<div class="footer">'
    'Knee MRI Multimodal AI • ResNet18 + TF-IDF + Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)
