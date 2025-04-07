import streamlit as st
import pandas as pd
import sklearn
from pycaret.classification import load_model, predict_model

# Force sklearn compatibility (critical for model loading)
sklearn.__version__ = "1.3.0"

# Set page config
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

@st.cache_resource
def load_wine_model():
    try:
        # Try loading with both .pkl extension and without
        try:
            return load_model('winequality')
        except:
            return load_model('winequality.pkl')
    except Exception as e:
        st.error(f"""
        **Model loading failed**: {str(e)}
        
        Common fixes:
        1. Ensure 'winequality.pkl' exists in your app directory
        2. Verify package versions match training environment
        3. Check the model wasn't corrupted during upload
        """)
        return None

def user_input_features():
    with st.expander("⚙️ Input Wine Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.4)
            volatile_acidity = st.slider('Volatile Acidity', 0.1, 2.0, 0.7)
            citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.0)
            residual_sugar = st.slider('Residual Sugar', 0.5, 20.0, 1.9)
            chlorides = st.slider('Chlorides', 0.01, 0.5, 0.076)
        
        with col2:
            free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1, 100, 11)
            total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 5, 300, 34)
            density = st.slider('Density', 0.98, 1.05, 0.9978)
            pH = st.slider('pH', 2.5, 4.5, 3.51)
            sulphates = st.slider('Sulphates', 0.3, 2.0, 0.56)
            alcohol = st.slider('Alcohol (%)', 8.0, 15.0, 9.4)
    
    return pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                        pH, sulphates, alcohol]],
                      columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                               'pH', 'sulphates', 'alcohol'])

def main():
    st.title('🍷 Wine Quality Predictor')
    st.markdown("""
    Predict wine quality (0-10 scale) based on physicochemical properties.
    """)
    
    # Load model
    model = load_wine_model()
    if model is None:
        st.stop()
    
    # Get user input
    input_df = user_input_features()
    
    # Display input
    st.subheader("Your Input Parameters")
    st.dataframe(input_df, hide_index=True)
    
    # Prediction
    if st.button("Predict Quality", type="primary"):
        with st.spinner('Analyzing wine characteristics...'):
            try:
                prediction = predict_model(model, data=input_df)
                quality = int(prediction['prediction_label'][0])
                
                # Visualize output
                st.subheader("Prediction Result")
                st.metric("Predicted Quality", f"{quality}/10")
                
                # Quality interpretation
                if quality >= 7:
                    st.success("🍾 High Quality Wine!")
                elif quality >=5:
                    st.info("🍷 Good Quality Wine")
                else:
                    st.warning("🫗 Below Average Quality")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()
