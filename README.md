# Multi-Modal Retrieval: Ensemble Embeddings & Diversity-Aware Search

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidDeez/multi-modal-retrieval-system/blob/main/notebooks/01_ensemble_benchmarking.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/yourusername/multi-modal-retrieval)

Research project on ensemble embeddings and diversity-aware search for robust multi-modal retrieval.

## 🎯 Key Features

- **Ensemble Embeddings**: CLIP + Sentence Transformers for improved retrieval robustness
- **Diversity-Aware FAISS**: Configurable similarity thresholds reducing redundancy by 40%
- **UMAP Visualization**: High-dimensional embedding analysis and interpretability
- **Real-time Analytics**: Full-stack system with sub-second latency on 1,000+ images

## 📊 Performance Metrics

- **Text Queries**: 0.82 mean similarity score
- **Image Queries**: 0.87 mean similarity score  
- **Diversity Improvement**: 40% reduction in redundant results
- **Latency**: Sub-second retrieval on 1,000+ image dataset

## 🚀 Quick Start

```bash
git clone https://github.com/DavidDeez/multi-modal-retrieval-system.git
cd multi-modal-retrieval-system
pip install -r requirements.txt

# Launch demo
python app.py
