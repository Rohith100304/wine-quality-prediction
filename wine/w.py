import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load model and data
@st.cache_resource
def load_wine_model():
    return load_model('wine/wine.pkl')

@st.cache_data
def load_dataset():
    return pd.read_csv('wine/winequality-red.csv')

model = load_wine_model()
data = load_dataset()

# User Input Form
def get_user_input():
    st.header("Wine Sample Quality Input")

    cols = st.columns(3)
    fixed_acidity = cols[0].number_input('Fixed Acidity', 4.0, 16.0, 7.4)
    volatile_acidity = cols[1].number_input('Volatile Acidity', 0.1, 1.5, 0.7)
    citric_acid = cols[2].number_input('Citric Acid', 0.0, 1.0, 0.0)

    cols2 = st.columns(3)
    residual_sugar = cols2[0].number_input('Residual Sugar', 0.5, 15.0, 1.9)
    chlorides = cols2[1].number_input('Chlorides', 0.01, 0.2, 0.076)
    free_sulfur_dioxide = cols2[2].number_input('Free Sulfur Dioxide', 1, 72, 11)

    cols3 = st.columns(3)
    total_sulfur_dioxide = cols3[0].number_input('Total Sulfur Dioxide', 6, 300, 34)
    density = cols3[1].number_input('Density', 0.990, 1.004, 0.9978)
    pH = cols3[2].number_input('pH', 2.5, 4.5, 3.51)

    cols4 = st.columns(2)
    sulphates = cols4[0].number_input('Sulphates', 0.3, 2.0, 0.56)
    alcohol = cols4[1].number_input('Alcohol', 8.0, 15.0, 9.4)

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

    return pd.DataFrame(user_data, index=[0])

# Main App
def main():
    st.title('Wine Quality Prediction App 🍷')
    st.write("This app predicts the quality of red wine based on its physicochemical properties.")

    st.sidebar.title("Options")

    if st.sidebar.button("View Dataset"):
        st.subheader("Wine Quality Dataset")
        st.write(data)

    # Download buttons
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='winequality-red.csv',
        mime='text/csv'
    )

    with open('wine/wine.pkl', 'rb') as f:
        model_bytes = f.read()

    st.sidebar.download_button(
        label="Download Model",
        data=model_bytes,
        file_name='winequality_model.pkl',
        mime='application/octet-stream'
    )

    # User input
    user_input = get_user_input()
    st.subheader("Wine Sample Summary")
    st.write(user_input)

    with st.expander("Wine Quality Scale Reference"):
            st.write("""
            - 3: Very Poor
            - 4: Poor
            - 5: Average
            - 6: Good
            - 7: Very Good
            - 8: Excellent
            """)
            
    if st.button("Predict Wine Quality"):
        prediction = predict_model(model, data=user_input)
        predicted_quality = prediction['prediction_label'][0]
        score = prediction['prediction_score'][0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Wine Quality: **{predicted_quality}** (Confidence: {score:.2%})")
    

if __name__ == '__main__':
    main()
