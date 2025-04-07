import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

@st.cache_resource
def load_wine_model():
    try:
        model = load_model('winequality')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title('🍷 Wine Quality Predictor')
    st.write('This app predicts wine quality based on physicochemical properties')

    # Load model
    model = load_wine_model()
    if model is None:
        st.error("Failed to load prediction model. Please check the model file.")
        return

    # Input form
    with st.form("prediction_form"):
        st.header("Input Wine Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_acidity = st.number_input('Fixed Acidity', min_value=4.0, max_value=16.0, value=7.0)
            volatile_acidity = st.number_input('Volatile Acidity', min_value=0.1, max_value=2.0, value=0.5)
            citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.3)
            residual_sugar = st.number_input('Residual Sugar', min_value=0.5, max_value=20.0, value=2.0)
            chlorides = st.number_input('Chlorides', min_value=0.01, max_value=0.5, value=0.08)
        
        with col2:
            free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=1, max_value=100, value=30)
            total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=5, max_value=300, value=100)
            density = st.number_input('Density', min_value=0.98, max_value=1.05, value=0.99)
            pH = st.number_input('pH', min_value=2.5, max_value=4.5, value=3.2)
            sulphates = st.number_input('Sulphates', min_value=0.3, max_value=2.0, value=0.6)
            alcohol = st.number_input('Alcohol (%)', min_value=8.0, max_value=15.0, value=10.5)
        
        submitted = st.form_submit_button("Predict Quality")
    
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                  chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                  pH, sulphates, alcohol]],
                                columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                         'pH', 'sulphates', 'alcohol'])
        
        # Make prediction
        try:
            prediction = predict_model(model, data=input_data)
            quality = prediction['prediction_label'][0]
            st.success(f'Predicted Wine Quality: {quality}')
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()
