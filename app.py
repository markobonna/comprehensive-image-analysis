import streamlit as st
from PIL import Image
from deepface import DeepFace
from colorthief import ColorThief
import matplotlib.pyplot as plt
import numpy as np

st.title("Comprehensive Image Analysis Web App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Resize the image to a smaller size to speed up processing
    image = image.resize((800, 800))

    try:
        # Analyze emotions using DeepFace
        st.write("Analyzing emotions...")
        result = DeepFace.analyze(np.array(image), actions=['emotion'], enforce_detection=False)
        
        # Access the correct keys from the result
        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']

        st.write(f"Dominant Emotion: {dominant_emotion}")
        st.write(f"Emotion Scores: {emotion_scores}")

        # Display emotion scores as a bar chart
        fig, ax = plt.subplots()
        ax.bar(emotion_scores.keys(), emotion_scores.values())
        st.pyplot(fig)

        # Analyze colors using ColorThief
        st.write("Analyzing colors...")
        image.save("temp_image.jpg")
        color_thief = ColorThief("temp_image.jpg")
        palette = color_thief.get_palette(color_count=6)
        st.write(f"Dominant Color: {palette[0]}")
        st.write(f"Color Palette: {palette}")

        # Display color palette
        fig, ax = plt.subplots()
        for i, color in enumerate(palette):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=[c/255 for c in color]))
        ax.set_xlim(0, len(palette))
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)

        # Predict popularity (for demonstration, we'll use a simple rule-based approach)
        st.write("Predicting popularity...")
        popularity_score = (emotion_scores['happy'] * 0.4 + emotion_scores['neutral'] * 0.3) * (1 - np.mean([np.abs(c[0]-128)/128 for c in palette]))
        st.write(f"Predicted Popularity Score: {popularity_score:.2f} (Higher is better)")

    except Exception as e:
        st.write("Error occurred during analysis:")
        st.write(str(e))
