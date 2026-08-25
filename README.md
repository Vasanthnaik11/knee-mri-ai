# 🦴 Knee MRI AI Analyzer

## Multimodal Knee Abnormality Detection using MRI + Radiology Report Text

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)

---

## 🚀 Live Demo

🌐 **Live Application:**  
https://knee-mri-ai-egm57ypsezrgmznb6rgc7f.streamlit.app/

The application combines processed knee MRI data with radiology report text to estimate probabilities for **12 knee abnormalities**.

### Features

- 🧲 Upload processed MRI volume (`.npy`)
- 📝 Enter a radiology report
- 🧠 Extract MRI image features using ResNet18
- 🔤 Process report text using TF-IDF
- 🤖 Generate predictions for 12 abnormalities
- 📊 Display probability scores
- 🔴 Highlight higher-probability findings
- 🟢/🔴 Display positive and negative predictions
- 🌐 Interactive Streamlit web application

---

# 📌 Project Overview

Knee MRI examinations contain a large amount of visual information that can be difficult to analyze manually.

This project explores a **multimodal AI approach** that combines:

1. 🧲 MRI image information
2. 📝 Radiology report text

The MRI component uses a **ResNet18-based encoder** to extract visual features, while the radiology report is represented using **TF-IDF**.

The system contains **12 machine-learning classifiers** that estimate probabilities for different knee abnormalities.

The project demonstrates how medical imaging and clinical text can be integrated into a single AI workflow.

---

# 🎯 Objectives

- Develop an AI-based knee MRI analysis pipeline.
- Extract visual features from MRI volumes.
- Process radiology reports using NLP.
- Combine MRI and text information.
- Predict multiple knee abnormalities.
- Build an interactive web application.
- Deploy the application using Streamlit.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │      USER INPUT     │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │    MRI Volume   │         │ Radiology Report│
        │      (.npy)     │         │      Text       │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ MRI Processing  │         │ Text Processing │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │    ResNet18     │         │      TF-IDF     │
        │  Image Encoder  │         │    Vectorizer   │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
          512 MRI Features             Text Features
                                             │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multimodal Model    │
                    │ / Feature Processing │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 12 ML Classifiers   │
                    │ Logistic Regression │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Probability        │
                    │ Predictions        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit App     │
                    └─────────────────────┘
🧬 Model Architecture
1. MRI Image Encoder

MRI volumes are processed using a ResNet18-based image encoder.

MRI Volume
    ↓
Preprocessing
    ↓
ResNet18
    ↓
512-dimensional MRI feature representation
MRI Features

512 features

2. Radiology Report Processing

The radiology report is processed using TF-IDF.

Radiology Report
       ↓
Text preprocessing
       ↓
TF-IDF Vectorizer
       ↓
1000-feature vocabulary
Text Features

TF-IDF vocabulary: 1000

3. Prediction

The multimodal pipeline uses trained machine-learning classifiers to estimate the probability of each abnormality.

Classifier

Logistic Regression

Number of Classifiers

12

🦴 Abnormalities Detected
#	Abnormality
1	ACL
2	MCL
3	Medial Meniscus
4	Lateral Meniscus
5	Medial OA
6	Lateral OA
7	PF OA
8	Effusion
9	Synovitis
10	Baker's
11	Contusion
12	Fracture
📊 Model Performance
Metric	Result
Macro AUC	0.7147
Abnormalities	12
MRI Encoder	ResNet18
MRI Features	512
Text Method	TF-IDF
Vocabulary	1000
Classifier	Logistic Regression
Macro AUC

0.7147

Macro AUC summarizes the model's ranking performance across the 12 abnormality prediction tasks.

🖥️ Streamlit Application

The application provides an interactive interface containing:

🧲 MRI Volume

Users upload a processed MRI volume:

.npy

Example:

knee_test_volume_small.npy

The processed test volume used during development had:

Shape: (186, 640, 640)
Dtype: float16
Size: approximately 145 MB
📝 Radiology Report

Users can paste a radiology report into the application.

🤖 AI Analysis

The application processes the inputs and produces probability estimates for all 12 abnormalities.

📋 Example Prediction

Example output from the application:

Abnormalities Analyzed: 12

Positive Predictions: 5

Model Macro AUC: 0.7147
Higher-Probability Findings
Medial Meniscus    100.00%
Baker's             99.93%
Fracture            99.82%
Contusion           99.21%
Medial OA            97.77%
Complete Example
ACL                 46.57%   NEGATIVE
MCL                 41.91%   NEGATIVE
Medial Meniscus    100.00%   POSITIVE
Lateral Meniscus     0.23%   NEGATIVE
Medial OA            97.77%   POSITIVE
Lateral OA            0.03%   NEGATIVE
PF OA                 0.14%   NEGATIVE
Effusion             12.15%   NEGATIVE
Synovitis             0.00%   NEGATIVE
Baker's              99.93%   POSITIVE
Contusion            99.21%   POSITIVE
Fracture             99.82%   POSITIVE

These values are example model outputs and are not clinical diagnoses.

📂 Repository Structure
knee-mri-ai/
│
├── app.py
├── README.md
├── requirements.txt
│
├── final_multimodal_models/
│
├── label_columns.pkl
├── logistic_models.pkl
├── text_selectors.pkl
├── tfidf_vectorizer.pkl
│
└── knee_mri_ai_final.zip
🛠️ Technologies Used
Programming
Python
Deep Learning
PyTorch
Torchvision
ResNet18
Machine Learning
Scikit-learn
Logistic Regression
TF-IDF
Data Processing
NumPy
Pandas
Pillow
Web Application
Streamlit
Deployment
GitHub
Streamlit Community Cloud
📦 Installation
1. Clone the Repository
git clone https://github.com/Vasanthnaik11/knee-mri-ai.git
cd knee-mri-ai
2. Create Virtual Environment
Windows
python -m venv venv

Activate:

venv\Scripts\activate
3. Install Dependencies
python -m pip install -r requirements.txt
▶️ Run Locally

Start the Streamlit application:

python -m streamlit run app.py

The application will be available at:

http://localhost:8501
📥 Input Data
MRI Input

The application accepts a processed MRI volume in:

.npy

format.

Example:

knee_test_volume_small.npy
Radiology Report

Paste the relevant radiology report into the text input field.

🔬 Dataset

This project uses the:

RSNA Knee Abnormality Detection Dataset

The dataset contains knee MRI studies organized into studies, series, and DICOM images.

Example MRI series include:

AX PD FS MPR L
COR PD L
SAG PD L
AX T2 SPIR L
SAG 3D VIEW PD SPAIR HR L

The original data contains DICOM MRI slices which are processed into numerical MRI volumes for model inference.

📊 Example MRI Study Structure

A knee MRI study can contain multiple imaging series.

Study
│
├── Series 1
│   └── 186 DICOM slices
│
├── Series 2
│   └── 25 DICOM slices
│
├── Series 3
│   └── 26 DICOM slices
│
├── Series 4
│   └── 30 DICOM slices
│
└── Series 5
    └── 320 DICOM slices

This demonstrates the multi-series structure of the MRI dataset.

🔐 Trained Model Files

The application uses trained model artifacts including:

label_columns.pkl
logistic_models.pkl
text_selectors.pkl
tfidf_vectorizer.pkl

These files are required for inference.

🚀 Deployment

The application is deployed using Streamlit Community Cloud.

Deployment Pipeline
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Install requirements
        ↓
Load trained models
        ↓
Start Streamlit
        ↓
Live Web Application
Live URL

https://knee-mri-ai-egm57ypsezrgmznb6rgc7f.streamlit.app/

📈 Future Improvements
🔹 Better MRI Models

Future versions could explore:

ResNet50
EfficientNet
DenseNet
Vision Transformers
3D CNNs
🔹 Better NLP

TF-IDF could be replaced or enhanced with:

ClinicalBERT
BioBERT
PubMedBERT
Sentence Transformers
🔹 Multimodal Fusion

Future versions could investigate:

Attention-based fusion
Cross-modal transformers
Joint embeddings
Late fusion
🔹 Explainable AI

Potential additions:

Grad-CAM
MRI slice visualization
Feature importance
Prediction explanations
🔹 Improved Evaluation

Future experiments could include:

Patient-level splitting
Cross-validation
Per-class AUC
Precision
Recall
F1-score
Calibration analysis
External validation
⚠️ Medical Disclaimer

IMPORTANT

This project is an AI research prototype.

It is NOT a medical diagnostic system and has not been approved for clinical use.

The predictions generated by this application are probabilistic estimates and may contain errors.

They should NOT be used as a substitute for:

Professional medical diagnosis
Radiologist interpretation
Physician evaluation
Clinical decision-making
Emergency medical care

Always consult a qualified healthcare professional for medical interpretation.

👨‍💻 Author
V. Vasanth Naik

B.Tech Engineering Graduate | AI & Data Science Enthusiast

Interests
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
Natural Language Processing
Data Science
🔗 Profiles

GitHub:
https://github.com/Vasanthnaik11

LinkedIn:
https://www.linkedin.com/in/vislavath-vathan-naik-42218625b

Email:
vislavathvasanthnaik@gmail.com

⭐ Project Status

🟢 LIVE AND DEPLOYED

Project: Knee MRI AI Analyzer

Type: Multimodal AI Research Prototype

MRI Model: ResNet18

MRI Features: 512

Text Model: TF-IDF

Vocabulary: 1000

Classifiers: 12

Macro AUC: 0.7147

Deployment: Streamlit Community Cloud
⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

📜 Acknowledgements

This project was developed for educational and research purposes using publicly available medical imaging resources and open-source machine-learning technologies.

Special thanks to the organizations and researchers who make medical imaging datasets and open-source AI tools available for research and education.
