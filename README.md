# End_to_End CRM_Support_System_app

<p align="center">
  <img src="https://www.xentric360.com/igrozech/2023/03/20201124_Blog-Featured-Image.gif" 
       alt="App Screenshot" 
       width="400">
</p>


# Customer Support Ticket Classification – Machine Learning & NLP Project

## 📌 Project Overview
This project focuses on building a **Machine Learning model** to classify customer support tickets into predefined categories based on the issue description.  
The goal is to help support teams prioritize and route tickets efficiently using NLP techniques.

---

## 📂 Dataset
- **Total records:** 24,000 customer support tickets  
- **Data type:** Text data (issue descriptions)
- **Target:** Ticket category / issue type

---

## ⚙️ Machine Learning Approach

### 1. Data Preprocessing
- Removed null and duplicate entries
- Converted text to lowercase
- Removed punctuation and stopwords
- Applied tokenization and vectorization

### 2. Feature Engineering
- Used **TF-IDF Vectorization** to convert text into numerical features

### 3. Model Used
- **Logistic Regression** (baseline classifier for text classification)

---

## 📊 Model Performance

| Metric        | Score |
|--------------|-------|
| Accuracy     | 89%   |
| Precision    | 88%   |
| Recall       | 87%   |
| F1-Score     | 87%   |

> Metrics are calculated on the test dataset to evaluate model performance.

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
 

## ✅ Key Takeaways
- Demonstrates end-to-end **text classification pipeline**
- Handles large-scale text data (24k rows)
- Suitable for real-world customer support automation use cases


## 🖥️ Application Features

### 👤 Customer Dashboard
- Enter customer name
- Submit support ticket
- Input validation with success feedback

### 🧑‍💼 Support Team Dashboard
- Classify tickets using ML model
- View predicted category with confidence score
- Track recently classified tickets
- Visualize ticket distribution


## 🚀 Live Demo
🔗 Click here to try the app:  
https://crmsupport-systemapp-chkxqbnqzpgtrcsgkvrc4v.streamlit.app/


**Author:** Vanashree G. Hegde👩🏻‍🦰 

