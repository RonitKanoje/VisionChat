# 🚀 VisionChat – AI-Powered Visual Intelligence System

VisionChat is a Google Lens–inspired AI application that combines deep learning–based image classification with Large Language Model (LLM) reasoning to generate contextual explanations for detected objects in images.

The system uses **VGG16 (PyTorch)** for object recognition and **Llama 3.2 (via Ollama)** to provide human-readable semantic insights based on model predictions.

---

## 🧠 Key Features

- Image classification using VGG16 (PyTorch implementation)
- Fine-tuned on a curated subset of the COCO dataset
- Context-aware explanations powered by Llama 3.2 (Ollama)
- Structured LLM workflow orchestration using LangGraph
- Observability and tracing with LangSmith
---

## 🖼️ Model Details

### VGG16 (PyTorch)

- Convolutional Neural Network with 13 convolution layers and 3 fully connected layers
- Fine-tuned on selected COCO dataset classes
- Used for object classification and feature extraction
- Predicted class labels passed to LLM for semantic interpretation

---

## 🤖 LLM Integration

VisionChat integrates **Llama 3.2 via Ollama** to:

- Generate contextual explanations of detected objects
- Provide real-world applications and insights
- Enhance interpretability beyond raw classification results
- Deliver human-readable AI responses

LangGraph manages structured LLM workflows and prompt orchestration.  
LangSmith is used for monitoring, tracing, and debugging LLM calls.

---

## 📂 Dataset

The project uses a curated subset of the **COCO (Common Objects in Context)** dataset from Kaggle.

Used for:

- Real-world object recognition training
- Improving robustness of classification
- Handling diverse contextual images


