# 🦿 Knee MRI Multimodal AI

An AI-powered research prototype for analyzing knee MRI volumes together with radiology report text to estimate the probability of multiple knee abnormalities.

The system combines ResNet18 image features, TF-IDF text features, and Logistic Regression models in a multimodal prediction pipeline and provides an interactive Streamlit web interface.

> ⚠️ **Medical Disclaimer:** This project is an AI research/portfolio prototype. Its predictions are probabilistic estimates and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

## 🚀 Live Demo

**Streamlit App:**  
https://knee-mri-ai-egm57ypsezrgmznb6rgc7f.streamlit.app/

## 📌 Project Overview

This project explores a multimodal AI approach for knee MRI abnormality analysis by combining:

- 🧲 MRI image information
- 📝 Radiology report text
- 🤖 Deep-learning image features
- 📊 Machine-learning text features
- 📈 Multi-label abnormality prediction

The system analyzes 12 different knee abnormalities and returns probability estimates for each one.

## 🧠 Model Architecture

MRI Volume
↓
MRI Preprocessing
↓
ResNet18
↓
512-D Image Features

Radiology Report
↓
TF-IDF
↓
Text Features

Image + Text Features
↓
Logistic Regression
↓
12 Abnormality Predictions
↓
Streamlit Web Application

## 🔬 Technologies Used

- Python
- PyTorch
- ResNet18
- Scikit-learn
- Logistic Regression
- TF-IDF
- NumPy
- DICOM
- Streamlit

## 📊 Model Information

| Component | Details |
|---|---|
| Image Model | ResNet18 |
| MRI Feature Size | 512 |
| Text Representation | TF-IDF |
| Prediction Model | Logistic Regression |
| Abnormality Classes | 12 |
| Validation Macro AUC | **0.7147** |

## 🩻 Abnormalities Detected

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

## 📈 Evaluation

The current validation result is:

### Macro AUC: 0.7147

The application provides probability estimates for all 12 abnormality classes.

## 🖥️ Application Features

### MRI Volume Upload

The application accepts a processed knee MRI volume in `.npy` format.

### Radiology Report Input

Users can paste radiology report text into the application.

### Multimodal Analysis

The application combines MRI and text information to generate abnormality probabilities.

### Prediction Results

The UI displays:

- Number of abnormalities analyzed
- Positive predictions
- Model Macro AUC
- Higher-probability findings
- Complete abnormality predictions

## 📂 Project Structure

```text
knee-mri-ai/
│
├── app.py
├── requirements.txt
│
└── final_multimodal_models/
    ├── tfidf_vectorizer.pkl
    ├── text_selectors.pkl
    ├── label_columns.pkl
    └── logistic_models.pkl
