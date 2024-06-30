Bangalore Restaurants Success Predictor

Visit the app: bangalore-restaurants-success-predictor-project-g7fhutiufsehbu.streamlit.app
Visit the app: [bangalore-restaurants-success-predictor](bangalore-restaurants-success-predictor-project-g7fhutiufsehbu.streamlit.app)
This project aims to predict the success of restaurants in Bangalore based on various features like online order availability, table booking options, location, votes, and more. The project involves exploratory data analysis (EDA) and machine learning model development using a range of classification algorithms.
Project Structure

- Bangalore Restaurants Success Predictor-EDA.ipynb: This notebook contains the exploratory data analysis (EDA) where various features of the dataset are analyzed and visualized.
- Bangalore Restaurants Success Predictor-Classification.ipynb: This notebook contains the classification model development and evaluation. It includes data preprocessing, feature engineering, model training, and hyperparameter tuning.
- streamlit_app.py: A Streamlit app to provide a user-friendly interface for predicting the success of a restaurant based on user inputs.

Installation

1- Clone the repository:
git clone https://github.com/moatazsaad/Bangalore-Restaurants-Success-Predictor-Project

2- Navigate to the project directory:
cd bangalore-restaurants-success-predictor

3- Create a virtual environment:
python3 -m venv venv

4- Activate the virtual environment:
On Windows:
    venv\Scripts\activate
On macOS and Linux:
    source venv/bin/activate
    
5- Install the required packages:
    pip install -r requirements.txt

Usage
Exploratory Data Analysis

1- Open the Bangalore Restaurants Success Predictor-EDA.ipynb notebook in Jupyter or Colab.
2- Run the cells to load the data, perform EDA, and visualize various features.

Classification Model

1- Open the Bangalore Restaurants Success Predictor-Classification.ipynb notebook in Jupyter or Colab.
2- Run the cells to preprocess the data, train the classification models, perform hyperparameter tuning, and save the best model.

Streamlit Application

1- Run the Streamlit application:
    streamlit run streamlit_app.py
2- Open the provided URL in your browser to access the application.
3- Provide the restaurant details in the input form and get the success prediction.

Data:
The dataset used in this project is a collection of Bangalore restaurant data from Zomato. It includes features such as online order availability, table booking options, location, votes, approximate cost for two people, cuisines, restaurant types, and more.

Models:
The project compares the performance of various classification models:

    Logistic Regression
    Decision Tree Classifier
    Random Forest Classifier
    K-Neighbors Classifier
    XGBoost Classifier

The final model is chosen based on the best cross-validation accuracy.

Results:
The best performing model is saved as best_model.pkl and the input features are saved as input_features.pkl. These files are used in the Streamlit app to make predictions based on user inputs.

Conclusion:
This project provides insights into the factors that contribute to the success of restaurants in Bangalore and offers a predictive model to help new restaurant owners make data-driven decisions.
