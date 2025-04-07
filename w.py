import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import base64
import io

# Load the saved model and dataset
@st.cache_resource
def load_wine_model():
    return load_model('winequality')

@st.cache_data
def load_dataset():
    return pd.read_csv('winequality-red.csv')

model = load_wine_model()
data = load_dataset()

# File download functions
def download_dataset():
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def download_model():
    with open('winequality.pkl', 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'data:file/pkl;base64,{b64}'
    return href

# Create a function to get user input in main area
def get_user_input():
    st.header("Wine Characteristics Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fixed acidity
        fixed_acidity = st.number_input('Fixed Acidity (g/dm³)', 
                                      min_value=4.0, max_value=16.0, value=8.0, step=0.1)
        
        # Volatile acidity
        volatile_acidity = st.number_input('Volatile Acidity (g/dm³)', 
                                         min_value=0.1, max_value=2.0, value=0.5, step=0.01)
        
        # Citric acid
        citric_acid = st.number_input('Citric Acid (g/dm³)', 
                                     min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        
        # Residual sugar
        residual_sugar = st.number_input('Residual Sugar (g/dm³)', 
                                       min_value=0.5, max_value=16.0, value=2.0, step=0.1)
        
        # Chlorides
        chlorides = st.number_input('Chlorides (g/dm³)', 
                                  min_value=0.01, max_value=0.5, value=0.08, step=0.01)
    
    with col2:
        # Free sulfur dioxide
        free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide (mg/dm³)', 
                                            min_value=1, max_value=100, value=15)
        
        # Total sulfur dioxide
        total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide (mg/dm³)', 
                                             min_value=5, max_value=300, value=40)
        
        # Density
        density = st.number_input('Density (g/cm³)', 
                                min_value=0.98, max_value=1.01, value=0.996, step=0.001)
        
        # pH
        pH = st.number_input('pH', 
                           min_value=2.7, max_value=4.0, value=3.3, step=0.1)
        
        # Sulphates
        sulphates = st.number_input('Sulphates (g/dm³)', 
                                  min_value=0.3, max_value=2.0, value=0.6, step=0.1)
        
        # Alcohol
        alcohol = st.number_input('Alcohol (% by volume)', 
                                min_value=8.0, max_value=15.0, value=10.0, step=0.1)
    
    # Store a dictionary into a dataframe
    user_data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    # Title
    st.title('Wine Quality Prediction App')
    st.write("""
    This app predicts the quality of red wine based on its physicochemical properties.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # View Dataset button
    if st.sidebar.button("View Dataset"):
        st.subheader("Wine Quality Dataset")
        st.write(data)
    
    # Download Dataset button
    dataset_download = download_dataset()
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='winequality_dataset.csv',
        mime='text/csv'
    )
    
    # Download Model button
    with open('winequality.pkl', 'rb') as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Download Model",
        data=model_bytes,
        file_name='winequality_model.pkl',
        mime='application/octet-stream'
    )
    
    # Get user input in main area
    user_input = get_user_input()
    
    # Display user input
    st.subheader('Wine Characteristics Summary')
    st.write(user_input)
    
    # Prediction button
    if st.button('Predict Wine Quality'):
        # Make prediction
        prediction = predict_model(model, data=user_input)
        
        # Display prediction
        st.subheader('Prediction Result')
        prediction_value = prediction['prediction_label'][0]
        
        # Quality interpretation
        quality_scale = {
            3: "Very Poor",
            4: "Poor",
            5: "Average",
            6: "Good",
            7: "Very Good",
            8: "Excellent"
        }
        
        quality_description = quality_scale.get(prediction_value, "Unknown")
        
        st.success(f'**Predicted Wine Quality: {prediction_value} ({quality_description})**')
        
        with st.expander("Wine Quality Scale Reference"):
            st.write("""
            - 3: Very Poor
            - 4: Poor
            - 5: Average
            - 6: Good
            - 7: Very Good
            - 8: Excellent
            """)
            
        

if __name__ == '__main__':
    main()
