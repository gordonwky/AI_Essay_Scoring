# 📘 Automated Essay Scoring with AI Language Models

This project explores the potential of using AI language models for **automated essay scoring (AES)**, inspired by the paper *"Exploring the Potential of Using an AI Language Model for Automated Essay Scoring"*. The goal is to integrate **LLM-based grading** with **linguistic features** and build a hybrid scoring pipeline evaluated using robust metrics like **Quadratic Weighted Kappa (QWK)**.

---

## 🚀 Features

- 🔍 **LLM Essay Grader**: Uses GPT (via OpenAI API) to predict rubric-aligned scores (1–6 scale).
- 📊 **Linguistic Feature Integration**: Incorporates lexical richness measures (`mattr_50`, `mtld`, `ttr`).
- 🌲 **Random Forest Classifier**: Trained using both LLM predictions and linguistic features.
- 🔄 **GridSearchCV with QWK**: Hyperparameter tuning based on Quadratic Weighted Kappa.
- 📉 **Evaluation**: Confusion matrix, accuracy, and test-set QWK scoring.

---

## 🧠 How It Works

1. **Essay Input** → Sent to GPT with a custom grading rubric.
2. **GPT Score** → Stored alongside other linguistic features.
3. **Random Forest Model** → Trained on human vs GPT scores + features.
4. **Model Evaluation** → Uses QWK and confusion matrix to evaluate performance.

---

## 🛠️ Tech Stack

- OpenAI GPT (Chat API)
- Python (pandas, sklearn, matplotlib)
- Jupyter Notebooks (for analysis)
- Scikit-learn RandomForest + GridSearchCV

---

## ✅ Future Improvements

- Add multi-rubric support
- Handle multilingual essays
- Explore transformer-based fine-tuned models for AES
- Visualize rubric alignment with attention heatmaps

---

## 📜 Citation

Inspired by:  
*Ramesh Manuvinakurike, et al. (2023). "Exploring the Potential of Using an AI Language Model for Automated Essay Scoring."*

---

## 💬 Contact

For questions or collaboration, feel free to reach out!


