
import streamlit as st
import pandas as pd 
import joblib
import sklearn
import xgboost
import category_encoders

# Load the pre-trained model and input features
Inputs = joblib.load("input_features.pkl")
Model = joblib.load("best_model.pkl")

def prediction(online_order, book_table, votes, location, approx_cost, listed_in_type, listed_in_city, cuisines_counts, rest_type_counts):
    # Create a DataFrame with input features
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0, "online_order"] = online_order
    test_df.at[0, "book_table"] = book_table
    test_df.at[0, "votes"] = votes
    test_df.at[0, "location"] = location
    test_df.at[0, "rest_type_counts"] = rest_type_counts
    test_df.at[0, "approx_cost(for two people)"] = approx_cost
    test_df.at[0, "cuisines_counts"] = cuisines_counts
    test_df.at[0, "listed_in(type)"] = listed_in_type
    test_df.at[0, "listed_in(city)"] = listed_in_city

    result = Model.predict(test_df)[0]
    return result

def main():
    st.set_page_config(page_title="Bangalore Restaurants Success Predictor", page_icon=":fork_and_knife:", layout="centered")
    st.title("Bangalore Restaurants Success Predictor :fork_and_knife:")

    st.markdown("## Provide the restaurant details to predict its success")
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            online_order = st.selectbox("Online Ordering", ['Yes', 'No'])
            book_table = st.selectbox("Table Booking", ['Yes', 'No'])
            votes = st.slider("Votes", min_value=0, max_value=16832, value=0, step=1)
            rest_type_counts = st.selectbox("Number of Restaurant Types", [1, 2])
            cuisines_counts = st.selectbox("Number of Cuisines", [1, 2, 3, 4, 5, 6, 7, 8])
            listed_in_type = st.selectbox("Type", ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out', 'Drinks & nightlife', 'Pubs and bars'])
        
        with col2:
            location = st.selectbox("Location", ['Banashankari', 'Basavanagudi', 'Jayanagar', 'JP Nagar', 'Bannerghatta Road', 'BTM', 'Electronic City', 'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR', 'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road', 'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar', 'St. Marks Road', 'Cunningham Road', 'Commercial Street', 'Vasanth Nagar', 'Domlur', 'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar', 'Kammanahalli', 'Koramangala 6th Block', 'Brookefield', 'Koramangala 4th Block', 'Banaswadi', 'Kalyan Nagar', 'Malleshwaram', 'Rajajinagar', 'New BEL Road'])
            approx_cost = st.slider("Approximate Cost for Two Persons (INR)", min_value=40, max_value=6000, value=500, step=10)
            listed_in_city = st.selectbox("City", ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur', 'Brigade Road', 'Brookefield', 'BTM', 'Church Street', 'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar', 'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli', 'Koramangala 4th Block', 'Koramangala 5th Block', 'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road', 'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road', 'Old Airport Road', 'Rajajinagar', 'Residency Road', 'Sarjapur Road', 'Whitefield'])
            submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        results = prediction(online_order, book_table, votes, location, approx_cost, listed_in_type, listed_in_city, cuisines_counts, rest_type_counts)
        label = ["Unsuccessful", "Successful"]
        st.success(f"The restaurant will be {label[results]}.")

if __name__ == '__main__':
    main()
