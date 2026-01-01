# Painkiller Recovery Predictor

An AI-powered dashboard that predicts patient recovery rates based on **painkiller dosage** using polynomial regression.  
Built with Flask, scikit-learn, and Chart.js, the app provides interactive visualizations and dosage insights.

---

## Features
- Polynomial Regression Model trained on dosage vs recovery dataset
- Interactive Dashboard built with Flask
- Chart.js Visualization with shaded underdose/overdose zones
- Prediction History table for tracking multiple inputs
- Professional UI/UX with responsive design

---

## Project Structure
medicine_recovery/
├── app.py                   # Flask app
├── model.py                 # ML utility functions
├── train_model.py          # Script to train and save model
├── medicine_recovery_dataset.csv  # Dataset
├── model.pkl                # Saved regression model
├── poly_transform.pkl      # Saved polynomial transformer
├── templates/
│   └── index.html           # Frontend HTML
├── static/
│   └── style.css            # Frontend CSS
└── README.md

Run the Flask app:
python app.py

Open in browser:
http://127.0.0.1:5000/

Requirements

Python 3.9+
Flask
scikit-learn
matplotlib
pandas
numpy
chart.js  (frontend)
