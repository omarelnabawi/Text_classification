import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your model and tokenizer
def load_model_and_tokenizer():
    model = load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to preprocess text and make predictions
def classify_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    
    # Check if sequences is empty
    if not sequences:
        return None

    # Pad sequences to the same length as used during training
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Replace your_max_length with the correct value

    # Predict the class of the input text
    prediction = model.predict(padded_sequences)
    return np.argmax(prediction)  # Adjust based on your classification

# Streamlit App UI
st.title("Text Classifier: Emotion Detection")

# Text input from the user
user_input = st.text_input("Enter a text to analyze the feeling:")

if st.button("Analyze"):
    if user_input:
        # Predict the feeling
        predicted_class = classify_text(user_input)
        
        if predicted_class is not None:
            # Map the prediction to feelings
            feelings = {4: 'sadness', 0: 'anger', 3: 'love', 5: 'surprise', 1: 'fear', 2: 'joy'}  # Adjust based on your model's classes
            result = feelings.get(predicted_class, "Unknown")

            # Display the result
            st.write(f"The feeling of this text is: **{result}**")
        else:
            st.write("Error: Unable to process the text.")
    else:
        st.write("Please enter some text to analyze.")
