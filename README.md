# **Bangalore Restaurants Success Predictor**

**Visit the app:** [bangalore-restaurants-success-predictor](bangalore-restaurants-success-predictor-project-g7fhutiufsehbu.streamlit.app)

This project aims to predict the success of restaurants in Bangalore based on various features like online order availability, table booking options, location, votes, and more. The project involves exploratory data analysis (EDA) and machine learning model development using a range of classification algorithms.

## **Project Structure**

- **EDA Notebook**: Analyzes and visualizes dataset features.
- **Classification Notebook**: Develops and evaluates classification models.
- **Streamlit App**: User interface for predicting restaurant success.

## **Installation**

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
The choice to use lowercase in the cd bangalore-restaurants-success-predictor command was based on the assumption that the directory name is in lowercase. This is a common convention in many projects to avoid issues with case sensitivity across different operating systems. If the actual directory name includes uppercase letters, it should be adjusted accordingly to match the actual directory structure. Here's the updated README with proper casing if needed:
Bangalore Restaurants Success Predictor

Visit the app: bangalore-restaurants-success-predictor

This project predicts the success of restaurants in Bangalore using features like online orders, table booking options, location, votes, and more. It involves exploratory data analysis (EDA) and machine learning models.
Project Structure

    EDA Notebook: Analyzes and visualizes dataset features.
    Classification Notebook: Develops and evaluates classification models.
    Streamlit App: User interface for predicting restaurant success.

Installation

    Clone the repository:

    bash

git clone https://github.com/moatazsaad/Bangalore-Restaurants-Success-Predictor-Project

Navigate to the project directory:

bash

cd Bangalore-Restaurants-Success-Predictor-Project

Create a virtual environment:

bash

python3 -m venv venv

Activate the virtual environment:

    Windows: venv\Scripts\activate
    macOS/Linux: source venv/bin/activate

Install required packages:

bash

    pip install -r requirements.txt

Usage
Exploratory Data Analysis

    Open EDA.ipynb in Jupyter or Colab.
    Run cells to load data, perform EDA, and visualize features.

Classification Model

    Open Classification.ipynb in Jupyter or Colab.
    Run cells to preprocess data, train models, tune hyperparameters, and save the best model.

Streamlit Application

    Run the Streamlit app:

    bash

    streamlit run streamlit_app.py

    Open the provided URL in your browser.
    Enter restaurant details to get a success prediction.

Data

The dataset includes features like online order availability, table booking options, location, votes, cost, cuisines, and restaurant types from Zomato.

Models:
The project compares the performance of various classification models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Neighbors Classifier
- XGBoost Classifier

The best model is chosen based on the best cross-validation accuracy.

Results:
The best performing model is saved as best_model.pkl and the input features are saved as input_features.pkl. These files are used in the Streamlit app to make predictions based on user inputs.

Conclusion:
This project provides insights into the factors that contribute to the success of restaurants in Bangalore and offers a predictive model to help new restaurant owners make data-driven decisions.
