# 🫀 Comprehensive Heart Disease Analysis

A complete machine learning pipeline for analyzing, predicting, and visualizing heart disease risks using the UCI Heart Disease dataset.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/yourusername/heart-disease-analysis)

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline that includes:

- **Data Preprocessing & Cleaning** - Handling missing values, encoding, and scaling
- **Feature Selection** - Statistical methods (Chi-Square Test) and ML-based techniques (RFE, SelectKBest)
- **Dimensionality Reduction** - PCA to retain essential features
- **Supervised Learning** - Logistic Regression, Decision Trees, Random Forest, SVM
- **Unsupervised Learning** - K-Means and Hierarchical Clustering
- **Model Optimization** - Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- **Interactive UI** - Streamlit web application
- **Deployment** - Ngrok tunneling for global access

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-analysis.git
   cd heart-disease-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   **Option A: Streamlit Web App (Recommended)**
   ```bash
   streamlit run heart_disease_analysis.py
   ```
   Then open your browser to `http://localhost:8501`

   **Option B: Command Line Interface**
   ```bash
   python heart_disease_analysis.py
   ```

### 🌐 Global Deployment (Bonus)

To deploy your app globally using Ngrok:

1. **Get Ngrok Auth Token**
   - Sign up at [ngrok.com](https://ngrok.com)
   - Get your auth token from the dashboard

2. **Set Environment Variable**
   ```bash
   # Windows
   set NGROK_AUTH_TOKEN=your_token_here
   
   # Linux/Mac
   export NGROK_AUTH_TOKEN=your_token_here
   ```

3. **Start Streamlit and Click "Start Ngrok Tunnel"**
   - The app will provide a public URL
   - Share this URL to access your app from anywhere!

## 📊 Features

### Data Analysis
- **Dataset Overview** - Comprehensive statistics and distributions
- **Correlation Analysis** - Feature relationships and dependencies
- **Missing Value Handling** - Automatic data cleaning
- **Feature Engineering** - Advanced preprocessing techniques

### Machine Learning Models

#### Supervised Learning
- **Logistic Regression** - Linear classification with regularization
- **Decision Trees** - Interpretable tree-based classification
- **Random Forest** - Ensemble method with feature importance
- **Support Vector Machine** - Non-linear classification with kernels

#### Unsupervised Learning
- **K-Means Clustering** - Partition-based clustering
- **Hierarchical Clustering** - Tree-based clustering
- **Silhouette Analysis** - Clustering quality assessment

### Feature Selection & Dimensionality Reduction
- **Chi-Square Test** - Statistical feature selection
- **Recursive Feature Elimination (RFE)** - ML-based selection
- **SelectKBest** - Top-k feature selection
- **Principal Component Analysis (PCA)** - Dimensionality reduction

### Model Optimization
- **GridSearchCV** - Exhaustive hyperparameter search
- **RandomizedSearchCV** - Randomized hyperparameter search
- **Cross-Validation** - Robust model evaluation
- **Performance Metrics** - Accuracy, AUC, Precision, Recall, F1-Score

### Visualizations
- **Data Distribution Plots** - Histograms and pie charts
- **Correlation Heatmaps** - Feature relationship visualization
- **Model Comparison Charts** - Performance benchmarking
- **PCA Analysis** - Explained variance plots
- **Confusion Matrices** - Classification performance
- **ROC Curves** - Model discrimination ability

## 🎮 Usage

### Streamlit Web Interface

1. **Pipeline Controls** - Run individual steps or complete pipeline
2. **Interactive Visualizations** - Click buttons to generate plots
3. **Real-time Results** - View model performance instantly
4. **Parameter Tuning** - Adjust hyperparameters interactively

### Command Line Interface

Run the complete pipeline with default settings:
```bash
python heart_disease_analysis.py
```

## 📈 Results

The pipeline provides comprehensive analysis including:

- **Model Performance Comparison** - Accuracy and AUC scores
- **Feature Importance Rankings** - Most predictive features
- **Clustering Quality Metrics** - Silhouette scores
- **Dimensionality Reduction Analysis** - PCA explained variance
- **Hyperparameter Optimization Results** - Best parameter combinations

## 🛠️ Technical Stack

- **Programming Language:** Python 3.8+
- **Core Libraries:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Framework:** Streamlit
- **Deployment:** Ngrok
- **Development:** Jupyter Notebooks

## 📁 Project Structure

```
heart-disease-analysis/
├── heart_disease_analysis.py    # Main pipeline implementation
├── demo.py                      # Quick demo script
├── test_visualizations.py       # Visualization testing
├── test_confusion_matrices.py   # Confusion matrix testing
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── LICENSE                      # MIT License
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- Open source contributors

## 📞 Contact

- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repository if you found it helpful!**
