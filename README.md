
![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Heart%20Disease%20Prediction.jpg?raw=true)
# Heart Disease Prediction Model
## 1. Introduction
The Heart Disease Prediction Model is a machine learning project that predicts the likelihood of heart disease based on various health indicators. Heart disease is one of the leading causes of death worldwide, making early detection crucial for preventive care. By analyzing patient data, this model aims to help healthcare providers identify high-risk patients, ultimately improving healthcare outcomes.

<div style="display: flex; justify-content: space-around; gap: 10px;">
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Heart%20Image%20GitHub%202.jpg?raw=true" alt="Heart Image 1" width="250" height="250" style="object-fit: cover;"/>
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Heart%20Image%20GitHub.jpg?raw=true" alt="Heart Image 2" width="250" height="250" style="object-fit: cover;"/>
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Stethescope%20Image%20GitHub.jpg?raw=true" alt="Stethoscope Image" width="250" height="250" style="object-fit: cover;"/>
</div>



It is also important that a doctor be present so that they could treat them. To make things worse, the tests usually take a long time before diagnosing whether a person would suffer from heart disease. However, it would be handy and better if we automate this process which ensures that we save a lot of time and effort on the part of the doctor and patient.

## 2. Machine Learning and Data Science
Machine learning and data science techniques are applied in this project to build predictive models from medical data. We explore different machine learning algorithms and data science practices, including data preprocessing, feature engineering, and model evaluation, to create a model capable of accurate heart disease risk prediction. The project uses Python, Scikit-learn, and TensorFlow, along with libraries like Pandas and Matplotlib for data handling and visualization.

## 3. Data
The dataset used for this project contains various medical features such as age, gender, blood pressure, cholesterol levels, and other health-related metrics. Each record represents a patient, with columns detailing their health metrics and a target variable indicating the presence or absence of heart disease. Key steps in data processing include:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## 4. Exploratory Data Analysis (EDA)
EDA helps uncover patterns and insights within the data, facilitating better feature selection and model building. EDA in this project includes:
- Statistical summary of data
- Distribution analysis of key health metrics
- Analysis of correlations between features and the target variable
- Identifying outliers and anomalies

This step lays the foundation for informed data preprocessing and model selection.

## 5. Visualizations
In this section, we will be visualizing some interests plots and trends in the data which is used by machine learning models to make predictions. We will also evaluate the performance of different machine learning and deep learning models on a given dataset by comparing various metrics. To identify the best version of each model, we will examine their hyperparameters.

To begin, we will review the list of rows and columns (features) in our dataset, which includes age, sex, cp (chest pain), chol, and others. This will provide insight into the types of features present and help us determine if additional features are necessary for analysis.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20213338.png)

Next, we will review the dataset for completeness by examining non-null values across each feature. With 303 entries, we confirm the dataset includes information on 1025 patients. Memory usage is minimal, so additional memory optimization steps, such as downcasting, are not needed.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20214418.png)

## 6. Machine Learning Models
A range of machine learning models were implemented to determine which performs best for this problem. The models include:
- **Logistic Regression**: A baseline model that predicts the probability of heart disease presence.
- **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies patients based on the similarity of their health metrics.
- **Decision Tree**: A model that segments data based on feature values, useful for interpretability.
- **Random Forest**: An ensemble model that improves accuracy by aggregating multiple decision trees.
- **Neural Network (TensorFlow)**: A deep learning model designed to capture complex relationships between features.

Each model was evaluated using accuracy, precision, recall, and F1-score, helping identify the best-performing model for heart disease prediction.

## 7. Outcomes
The models achieved over 80% accuracy in predicting heart disease risk, with the Random Forest and Neural Network models showing the best performance. Key outcomes include:
- **Model Accuracy**: Overall accuracy achieved in distinguishing high-risk patients.
- **Feature Importance**: Analysis of the most important features contributing to heart disease, such as cholesterol levels and age.
- **Evaluation Metrics**: Detailed evaluation of precision, recall, and F1-score for each model, enabling a balanced assessment of model strengths and weaknesses.

## 8. Future Scope
Future enhancements for this project include:
- **Additional Data**: Incorporating more patient data from diverse demographics to improve model generalizability.
- **Advanced Feature Engineering**: Exploring derived features, such as lifestyle factors or medication history, for deeper insights.
- **Hyperparameter Tuning**: Further fine-tuning model parameters to enhance accuracy.
- **Web Application Integration**: Developing a user-friendly interface to allow healthcare professionals to input data and receive risk predictions instantly.
- **Deployment**: Deploying the model as a web application or mobile app for broader accessibility.

