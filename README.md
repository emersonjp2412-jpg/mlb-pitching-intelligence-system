# ⚾ MLB Pitching Intelligence System

End-to-end MLB pitching analytics system combining biomechanics, Statcast data, clustering, and machine learning to model pitcher velocity and performance efficiency.

---

## 🧠 Project Objective

This project simulates MLB R&D workflows to analyze how pitchers generate velocity and how biomechanical + statistical features influence performance.

The goal is not only prediction, but **understanding pitcher archetypes and mechanical efficiency**.

---

## ⚾ Pitchers Included

- Gerrit Cole  
- Spencer Strider  
- Corbin Burnes  
- Justin Verlander  
- Zack Wheeler  
- Shane McClanahan  

---

## 🧬 Methodology

### 1. Biomechanical Feature Engineering
Engineered performance features inspired by pitching mechanics:

- Release efficiency  
- Movement profiles  
- Spin-related metrics  
- Velocity differentials  

---

### 2. Machine Learning Model
A supervised learning model was built to predict pitch velocity using:

- XGBoost regression  
- Feature importance analysis (SHAP)

---

### 3. Clustering Analysis
Unsupervised learning used to identify pitcher archetypes:

- UMAP (dimensionality reduction)  
- HDBSCAN (clustering)  

Result: grouping pitchers by mechanical similarity.

---

## 📊 Key Outputs

- Pitch velocity prediction model  
- Mechanical efficiency scoring system  
- Pitcher archetype clustering  
- Scouting-style insights per pitcher  
- Automated scouting report (PDF)

---

## 🧠 Key Insight

Velocity is not only a function of strength, but a combination of:

- mechanics  
- release efficiency  
- movement patterns  
- pitch design

This system helps translate raw Statcast data into actionable scouting intelligence.

---

## 🛠️ Tech Stack

- Python  
- pandas / numpy  
- XGBoost  
- SHAP  
- UMAP  
- HDBSCAN  
- pybaseball  

---

## 🚀 Future Work

- Injury risk modeling (Tommy John risk indicators)  
- Pitch tunneling analysis  
- Pitch sequencing prediction  
- Interactive scouting dashboard (Streamlit)  

---

## 📬 Purpose

Built as a sports analytics portfolio project focused on baseball performance analysis and MLB-style data science workflows.
