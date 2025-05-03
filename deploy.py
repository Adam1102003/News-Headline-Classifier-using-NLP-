import streamlit as st
import joblib
import pandas as pd

# Load models for each feature extraction method
models = {
    "Bag of Words": joblib.load("BOW_Stem_Tfidf_Models/BOW_Stem_Tfidf_ModelsSVM_bow_stem_model.pkl"),
    "POS Tags": joblib.load("POS_Spacy_Models/Neural Network_pos_spacy_model.pkl"),
    "NER": joblib.load("NER_New_Models/Neural Network_ner_new_model.pkl"),
    "NER + POS": joblib.load("NER_POS_Models/Neural Network_ner_pos_model.pkl")
}

# Preprocessing functions (replace with your actual implementations)
def preprocess_headline(headline, method):
    if method == "Bag of Words":
        # Preprocess for BoW
        pass
    elif method == "POS Tags":
        # Preprocess for POS
        pass
    elif method == "NER":
        # Preprocess for NER
        pass
    elif method == "NER + POS":
        # Preprocess for NER + POS
        pass
    return pd.DataFrame([headline])  # Example placeholder

# Streamlit app
st.title("News Headline Classifier")
st.markdown("Classify news headlines into categories based on your chosen feature extraction method.")

# Feature extraction method selection
feature_method = st.selectbox("Select Feature Extraction Method:", list(models.keys()))

# Headline input
headline = st.text_input("Enter a news headline:")

# Predict button
if st.button("Predict Category"):
    if headline.strip() == "":
        st.warning("Please enter a headline to classify.")
    else:
        with st.spinner("Classifying..."):
            # Preprocess the headline
            processed_headline = preprocess_headline(headline, feature_method)
            # Predict category
            model = models[feature_method]
            category = model.predict(processed_headline)[0]
            st.success(f"The predicted category is: **{category}**")
