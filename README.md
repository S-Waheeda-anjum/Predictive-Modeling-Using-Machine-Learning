# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview

This project predicts house prices using the **Ames Housing Dataset** and a **Random Forest Regressor** model. The project demonstrates a complete Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## 🎯 Objective

- Load and explore the housing dataset
- Handle missing values
- Encode categorical features
- Train a Machine Learning model
- Evaluate model performance
- Predict house prices
- Visualize important insights

---

## 📂 Dataset

- **Dataset Name:** Ames Housing Dataset
- **Rows:** 2930
- **Columns:** 82
- **Target Variable:** SalePrice

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Visual Studio Code

---

## 📚 Python Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

---

## 📁 Project Structure

```text
Predictive-Modeling-Using-Machine-Learning
│
├── dataset
│   ├── housing.csv
│   └── cleaned_housing.csv
│
├── images
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
│
├── models
│   └── house_price_model.pkl
│
├── notebooks
│   └── analysis.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluation.py
│   └── prediction.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🔄 Workflow

1. Load the dataset
2. Explore the dataset
3. Handle missing values
4. Encode categorical variables
5. Train the Random Forest model
6. Evaluate the model
7. Save the trained model
8. Predict house prices

---

## 📊 Model Performance

| Metric | Value |
|---------|---------|
| Algorithm | Random Forest Regressor |
| R² Score | **0.9102** |

### Performance Summary

The Random Forest model achieved an **R² Score of 0.9102**, indicating that it explains approximately **91% of the variation** in house prices. This demonstrates excellent predictive performance on the Ames Housing Dataset.

---

## 📈 Visualizations

The project includes the following visualizations:

- Actual vs Predicted Prices
- Residual Plot
- Feature Importance Chart

---

## 💡 Key Features

- Data Cleaning
- Missing Value Handling
- One-Hot Encoding
- Feature Engineering
- Random Forest Regression
- Model Evaluation
- House Price Prediction
- Data Visualization

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Predictive-Modeling-Using-Machine-Learning.git
```

### Navigate to the project

```bash
cd Predictive-Modeling-Using-Machine-Learning
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run preprocessing

```bash
python src/preprocessing.py
```

### Train the model

```bash
python src/train_model.py
```

### Evaluate the model

```bash
python src/evaluation.py
```

### Predict house price

```bash
python src/prediction.py
```

---

## 📷 Project Output

The project generates:

- Trained Machine Learning model
- Evaluation metrics
- Prediction results
- Visualization graphs

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Compare multiple regression algorithms
- Deploy the model using Flask or Streamlit
- Build an interactive web application
- Integrate real-time prediction functionality

---

## 👩‍💻 Author

**Shaik Waheeda Anjum**

M.Tech Computer Science Engineering Student

Data Science Enthusiast | Machine Learning Enthusiast

---

## ⭐ If you found this project helpful, consider giving it a Star!