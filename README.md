# Hybrid Crop Yield Prediction Using Ensemble Machine Learning

## Project Overview
This project shifts agricultural management from reactive to predictive by forecasting crop yields with high accuracy.By integrating diverse data sources—including historical meteorological records, soil nutrient profiles, and satellite imagery (NDVI)—the model identifies critical non-linear patterns to optimize resource allocation and stabilize food supply.

## Key Objectives & Results
***Predictive Accuracy:** Targeted a Coefficient of Determination ($R^2$) above 0.85 and utilized Root Mean Squared Error (RMSE) for rigorous evaluation.
***Multi-Source Integration:** Leveraged Weather APIs (temperature, rainfall), Satellite Imaging (NDVI from Sentinel-2/Landsat), and IoT soil sensors (pH, moisture).
* **Actionable Insights:** Identified the most significant environmental predictors (e.g., rainfall during flowering, soil nitrogen) to assist farmers and policymakers.

##  Technical Implementation
* **Feature Engineering:** Created crop-specific features such as "Total precipitation during the vegetative stage" and "Stress Days" (count of days exceeding 35°C).
* **Model Selection:** Benchmarked Multiple Linear Regression against advanced tree-based models like **Random Forest** and **XGBoost**, as well as deep learning **LSTMs** for temporal patterns.
* **Evaluation Strategy:** Implemented a **temporal split** (training on 2000–2018 data and testing on 2019–2020) to mimic real-world forecasting scenarios.

##  Tech Stack
* **Deep Learning/ML:** TensorFlow, PyTorch, Scikit-learn.
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn.
* **Deployment:** Designed as a web application/API for real-time field use.
