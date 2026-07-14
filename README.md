# 🌱 OptiCrop AI – Smart Agricultural Production Optimization Engine

> **An AI-powered web application that recommends the most suitable crop using soil nutrients and environmental conditions, enabling smarter and more sustainable agricultural decisions.**

---

## 📖 Project Overview

OptiCrop AI is an intelligent agricultural decision support system developed to help farmers, researchers, and agricultural enthusiasts make informed crop selection decisions. The application uses Machine Learning algorithms to analyze essential soil nutrients and climatic conditions, providing accurate crop recommendations based on scientific data.

The system evaluates parameters such as **Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, and Rainfall** to determine the most suitable crop for cultivation. In addition to crop prediction, the platform offers crop suitability analysis, agricultural analytics, prediction history, crop information, and downloadable reports through an interactive web interface.

---

## 🎯 Objectives

- Develop an intelligent crop recommendation system using Machine Learning.
- Assist farmers in selecting the most suitable crop based on soil and climate conditions.
- Improve agricultural productivity through data-driven recommendations.
- Perform data preprocessing and exploratory data analysis.
- Compare multiple Machine Learning algorithms to identify the best-performing model.
- Deploy the trained model through a Flask web application.
- Maintain user prediction history for future reference.
- Provide downloadable PDF prediction reports.
- Create a responsive, user-friendly, and secure web platform.

---

## ✨ Key Features

### 🔐 User Authentication

- User Registration
- Secure Login
- Logout Functionality
- Protected User Dashboard

### 🌾 Smart Crop Recommendation

Predicts the most suitable crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

### 🌱 Crop Suitability Assessment

Evaluates whether the selected crop is suitable under the provided environmental conditions and offers detailed suitability insights.

### 📊 Agricultural Analytics Dashboard

Provides visual insights through:

- Crop Distribution Charts
- Soil Nutrient Analysis
- Environmental Parameter Visualization
- Prediction Statistics
- Agricultural Trends

### 📚 Crop Information Library

Displays useful information about recommended crops, including:

- Crop Description
- Growing Season
- Productivity
- Cultivation Tips

### 🕒 Prediction History

Stores previous predictions made by authenticated users for easy reference.

### 📄 Download Prediction Report

Allows users to generate and download prediction reports in PDF format.

### 💬 Feedback System

Enables users to submit feedback and ratings to improve the application.

### 📱 Responsive Interface

Designed to work efficiently across desktops, tablets, and mobile devices.

---

## 🧠 Machine Learning Workflow

The project follows a structured Machine Learning pipeline:

1. Dataset Collection
2. Data Cleaning
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Selection
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Comparison
10. Model Serialization
11. Flask Application Integration
12. Web Deployment

---

## ⚙️ Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Backend

- Flask

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap

### Database

- SQLite

### Model Storage

- Joblib
- Pickle

### Development Tools

- VS Code
- Git
- GitHub

---

## 📂 Dataset

The application uses a Crop Recommendation Dataset containing agricultural and environmental parameters that help train the Machine Learning model.

### Dataset Attributes

| Attribute | Description |
|-----------|-------------|
| Nitrogen (N) | Nitrogen content in soil |
| Phosphorus (P) | Phosphorus content in soil |
| Potassium (K) | Potassium content in soil |
| Temperature | Atmospheric temperature |
| Humidity | Relative humidity |
| Soil pH | Acidity or alkalinity of soil |
| Rainfall | Annual rainfall |
| Crop Label | Recommended Crop |

---

## 🤖 Machine Learning Models

The following Machine Learning algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

---

## 🏆 Best Performing Model

### Random Forest Classifier

The Random Forest Classifier demonstrated the highest prediction accuracy and overall performance during model evaluation. Due to its robustness, reliability, and ability to handle complex agricultural datasets, it was selected as the final deployed model for crop prediction.

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarah-tahoorah/OptiCrop-Smart-Agricultural-Production-.git
```

### Step 2: Navigate to the Project Folder

```bash
cd OptiCrop-Smart-Agricultural-Production-
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

### Step 5: Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📁 Project Structure

```text
OptiCrop-Smart-Agricultural-Production-/
│
├── dataset/
├── models/
├── notebooks/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
├── instance/
├── app.py
├── auth.py
├── database.py
├── requirements.txt
├── README.md
├── opti_crop.db
└── assets/
```

---

## 🖥️ Application Modules

- 🏠 Home
- 🔐 Login & Registration
- 🌾 Crop Recommendation
- 🌱 Crop Suitability Assessment
- 📊 Agricultural Analytics Dashboard
- 📚 Crop Information Library
- 🕒 Prediction History
- 📄 PDF Report Generation
- 💬 Feedback

---

## 📈 Advantages

- Intelligent crop recommendations
- Faster agricultural decision-making
- Efficient use of soil resources
- Sustainable farming support
- Interactive data visualization
- Secure user authentication
- Historical prediction tracking
- Easy-to-use interface

---

## 🌍 Future Enhancements

- 🌦️ Live Weather API Integration
- 🌱 Fertilizer Recommendation System
- 🛰️ Satellite-Based Crop Monitoring
- 🤖 Deep Learning Models
- ☁️ Cloud Deployment
- 📱 Android & iOS Mobile Application
- 🌍 Multi-language Support
- 📍 GPS-Based Soil Analysis
- 📊 Advanced Business Intelligence Dashboard

---

## 👩‍💻 Developer

**Sarah Tahoorah**

**B.Tech – Computer Science & Engineering (Artificial Intelligence)**

**G. Pullaiah College of Engineering and Technology**

---

## 📜 License

This project is developed for educational and academic purposes. It may be extended or modified for research and learning in the field of Artificial Intelligence and Smart Agriculture.

---

## 🙏 Acknowledgements

Special thanks to the open-source community and the developers of **Python, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, Bootstrap, and SQLite** for providing the technologies that made this project possible.

---

## 📌 Conclusion

OptiCrop AI demonstrates the practical application of Artificial Intelligence and Machine Learning in modern agriculture by providing accurate crop recommendations based on soil nutrients and environmental conditions. The project integrates data preprocessing, model development, visualization, secure user authentication, prediction history, and an interactive web interface into a single intelligent platform.

By helping users make informed farming decisions, OptiCrop AI contributes to improving agricultural productivity, resource efficiency, and sustainable farming practices. With future enhancements such as weather integration, fertilizer recommendations, cloud deployment, and mobile accessibility, the system has the potential to evolve into a comprehensive smart agriculture solution.