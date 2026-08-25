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
