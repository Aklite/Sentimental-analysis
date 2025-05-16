# 📘 Decoding Emotion Through Sentiment Analysis of Social Media Conversation

## 🔍 Problem Statement

In today’s digital world, social media platforms have become a powerful medium for users to express thoughts, emotions, and opinions. These vast amounts of unstructured text data hold valuable emotional insights, but interpreting them manually is impractical. This project aims to build a machine learning model capable of decoding emotions such as **joy, anger, sadness, surprise, fear**, etc., from user-generated content across social platforms.

## 🎯 Project Objectives

- Classify social media text into emotional categories (Positive, Neutral, Negative).
- Clean and preprocess noisy, informal social media data.
- Perform EDA to uncover patterns and emotional trends.
- Engineer features using TF-IDF vectorization and categorical encoding.
- Train models: Logistic Regression, Naive Bayes, XGBoost, SVM.
- Evaluate using accuracy, precision, recall, and F1-score.
- Deploy an interactive web app using Streamlit.

## 📊 Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)
- **Records**: 732
- **Format**: CSV
- **Features**: `Text`, `Sentiment`, `Timestamp`, `User`, `Platform`, `Likes`, `Retweets`, `Country`, etc.
- **Classes**: Positive, Negative, Neutral

## 🛠 Tools & Technologies

- **Language**: Python 3.10+
- **Libraries**: pandas, numpy, matplotlib, seaborn, nltk, sklearn, xgboost, streamlit
- **Modeling**: Logistic Regression, Naive Bayes, XGBoost, SVM
- **Text Processing**: NLTK, TF-IDF, Stopword Removal, Lemmatization
- **Deployment**: Streamlit
- **IDE**: Google Colab, Jupyter
- **Version Control**: GitHub

## 🔁 Workflow

Data Collection → Preprocessing → EDA → Feature Engineering → Model Building → Evaluation → Deployment


## 📈 Model Performance

| Model               | Accuracy   | Highlights                             |
|---------------------|------------|----------------------------------------|
| Logistic Regression | ~85%       | Baseline model                         |
| Naive Bayes         | ~82%       | Simple, fast, but less precise         |
| XGBoost             | ~89%       | Best performance overall               |
| SVM                 | ~87%       | Robust with TF-IDF features            |

## 📊 Visualizations

- Sentiment class distribution
- Word clouds per sentiment
- Confusion matrix
- Classification report
- Feature importance (XGBoost)

## 🚀 Deployment

- **Platform**: Streamlit
- **URL**: _(https://sentimental-analysis-eckydwhon4i4hj6ewadmsv.streamlit.app/)_
- **How to run locally**:

```bash
streamlit run app.py

    Input: A sentence describing an emotion

    Output: Predicted Emotion class (e.g., Happy, Sad, Neutral)

🧠 Example Predictions
Input	Predicted Emotion
"I failed my exam so sad"	Sad 😢
"I won a prize in competition I'm happy !"	Happy 😊
"Just working on my tasks i was neutral."	Neutral 😐
[ the input was mention the key words to predict the analysis correctly ("because the trained model will be have  the less datas" )]

👨‍💻 Team Members
Name	Role
Arunkumar P	EDA, Feature Engineering, Deployment
Avinash N	Model Building & Evaluation
Arunagiri D	Data Collection & Preprocessing

🔮 Future Scope

    Integrate real-time Twitter data via Twitter API

    Include sarcasm/irony detection using BERT or GPT

    Expand to support multilingual emotion detection

    Live dashboard for emotional trend monitoring

📁 Repository Structure

📁 Sentiment-Analysis/
├── app.py                     # Streamlit application
├── emotion_model.pkl          # Trained model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── cleaned_sentiment_dataset.csv # Preprocessed data
├── README.md                  # Project description
└── requirements.txt           # Python dependencies
