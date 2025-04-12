# **Sentiment Analysis on Twitter Dataset**  
 **A Machine Learning & NLP Project for Sentiment Classification**  

---

##  **Project Overview**  
This project aims to analyze **Twitter sentiment** using the **Sentiment140 dataset** and various **Natural Language Processing (NLP) techniques**. We classify tweets into **positive, negative, or neutral** sentiments using machine learning and deep learning models.  

** Key Highlights:**  
- **Dataset:** Sentiment140 (1.6M labeled tweets)  
- **Techniques:** Text preprocessing, feature engineering (TF-IDF, Word2Vec, BERT)  
- **Models:** Logistic Regression, SVM, LSTM, BERT  
- **Evaluation:** Accuracy, F1-score, ROC-AUC  

---

##  **Project Objectives**  
 **Preprocess** raw text data for sentiment analysis  
 **Train & Compare** machine learning and deep learning models  
 **Evaluate** model performance using key metrics  
 **Visualize** sentiment trends & feature importance  
 **Provide insights** for real-world applications (e.g., business analytics, social media monitoring)  

---

##  **Repository Structure**  

```
📂 Sentiment-Analysis-Twitter
│── 📁 data                # Datasets (raw & processed)
│── 📁 notebooks           # Jupyter notebooks (EDA, training)
│── 📁 src                 # Python scripts (preprocessing, training, evaluation)
│── 📁 models              # Saved trained models
│── 📁 results             # Evaluation reports, graphs
│── 📁 docs                # Documentation files
│── 📁 tests               # Unit tests for model reliability
│── requirements.txt       # Python dependencies
│── README.md              # Project overview (this file)
│── .gitignore             # Ignored files (e.g., logs, cache)
│── LICENSE                # License information
```

---

##  **Dataset**  
We use the **Sentiment140 dataset**, which contains 1.6 million tweets labeled as **positive, negative, or neutral**. The dataset is publicly available and widely used for sentiment analysis research.

 **Dataset Features:**  
- `text`: Tweet content  
- `sentiment`: Label (0 = negative, 2 = neutral, 4 = positive)  
- `user`, `date`, `query`, `id`: Additional metadata  

 **Download**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  

---

##  **Installation & Setup**  

###  **1. Clone the Repository**  
```bash
git clone https://github.com/Northeastern-MSDAE/IE7500.git
cd IE7500
```

###  **2. Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

###  **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

###  **4. Run Data Preprocessing**  
```bash
python src/preprocessing.py
```

###  **5. Train the Model**  
```bash
python src/train.py
```

###  **6. Evaluate Model Performance**  
```bash
python src/evaluate.py
```

---

##  **Methodology**  

### **1. Data Preprocessing**  
- **Tokenization**: Splitting tweets into words  
- **Lemmatization/Stemming**: Reducing words to their root forms  
- **Stopword Removal**: Filtering out common words like "the", "is", etc.  
- **Handling Class Imbalance**: Oversampling or undersampling  

### **2. Feature Engineering**  
- **TF-IDF (Term Frequency-Inverse Document Frequency)**  
- **Word2Vec Embeddings**  
- **Pre-trained Embeddings (GloVe, BERT)**  

### **3. Model Development**  
- **Traditional ML Models**: Logistic Regression, Support Vector Machines  
- **Deep Learning**: LSTM, BERT  

### **4. Model Evaluation**  
- **Metrics Used**: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **Cross-validation** for better generalization  

---

##  **Results & Visualizations**  

 **Sentiment Distribution:**  
 **Confusion Matrix:**  
 **Feature Importance (TF-IDF Weights):**  


 **Best Model:** **update later**  

---

##  **Applications of Sentiment Analysis**  
 **Business Insights** → Track customer feedback & brand sentiment  
 **Social Media Monitoring** → Analyze trends & public opinions  
 **Recommendation Systems** → Improve product recommendations  
 **Finance & Market Analysis** → Predict stock trends based on sentiment  

---

##  **Contributing**  
We welcome contributions to improve this project!  

### **How to Contribute?**  
1. **Fork the Repository**  
2. **Create a Feature Branch**  
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Commit Your Changes**  
   ```bash
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```
4. **Open a Pull Request (PR) on GitHub**  

---

##  **License**  
This project is licensed under the **MITLicense** – feel free to use and modify it.  

---

##  **Contact & Support**  
For any issues or inquiries, please open an **Issue** on GitHub or reach out via:  

📧 **Email**: IE7500@northeastern.edu  
🌐 **Website**: [Sentiment Analysis Project](https://github.com/Northeastern-MSDAE/IE7500)  

 *If you found this project useful, don't forget to ⭐ the repo!*  

---
✔ **Last Updated:** March 2025  
🔗 **Repository Link:** [GitHub](https://github.com/Northeastern-MSDAE/IE7500)