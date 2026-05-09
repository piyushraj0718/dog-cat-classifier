import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import plotly.graph_objects as go
import time


# ======================================
# PAGE SETTINGS
# ======================================

st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide"
)


# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main {
    background-color: #050816;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    font-size: 58px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #9CA3AF;
    font-size: 18px;
    margin-bottom: 35px;
}

.stButton > button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    height: 3em;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

.creator {
    text-align: center;
    color: #9CA3AF;
    margin-top: 20px;
    font-size: 14px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ======================================
# HEADER
# ======================================

st.markdown(
    "<div class='main-title'>🐶 Dog vs Cat Classifier 🐱</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>A simple deep learning mini project using TensorFlow and Streamlit</div>",
    unsafe_allow_html=True
)


# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_classification_model():

    model = load_model("dog_cat_model1.keras")

    return model


try:

    model = load_classification_model()
    st.success("Model loaded successfully!")

except Exception as e:

    st.error(f"Error loading model: {e}")


# ======================================
# MAIN LAYOUT
# ======================================

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.1], gap="large")


# ======================================
# LEFT COLUMN
# ======================================

with left_col:

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a cat or dog image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    predict_button = False

    if uploaded_file is not None:

        uploaded_image = Image.open(uploaded_file)

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            width=350
        )

        st.write("")

        predict_button = st.button("🔍 Predict Image")


# ======================================
# RIGHT COLUMN
# ======================================

with right_col:

    st.subheader("📊 Prediction Result")

    if uploaded_file is not None and predict_button:

        with st.spinner("Analyzing image..."):

            time.sleep(1)

            # Image preprocessing
            processed_image = uploaded_image.resize((150, 150))

            image_array = image.img_to_array(processed_image)
            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )[0][0]

            dog_probability = float(prediction)
            cat_probability = float(1 - prediction)

            # Result logic
            if prediction > 0.5:

                final_result = "🐶 Dog"
                confidence = dog_probability

                st.success(
                    f"Prediction: Dog\n\nConfidence Score: {confidence:.2%}"
                )

            else:

                final_result = "🐱 Cat"
                confidence = cat_probability

                st.success(
                    f"Prediction: Cat\n\nConfidence Score: {confidence:.2%}"
                )

            st.write("### Confidence Scores")

            # Confidence bars
            st.write(f"🐱 Cat: {cat_probability:.2%}")
            st.progress(cat_probability)

            st.write(f"🐶 Dog: {dog_probability:.2%}")
            st.progress(dog_probability)

            # Plotly graph
            chart = go.Figure(

                data=[

                    go.Bar(

                        x=["Cat", "Dog"],

                        y=[
                            cat_probability,
                            dog_probability
                        ],

                        text=[
                            f"{cat_probability:.2%}",
                            f"{dog_probability:.2%}"
                        ],

                        textposition='auto'

                    )

                ]

            )

            chart.update_layout(

                template="plotly_dark",

                title="Prediction Probability",

                height=400,

                yaxis=dict(
                    range=[0,1]
                )

            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

    else:

        st.info("Upload an image and click Predict.")


# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.title("Project Info")

    st.markdown("""

    ### About

    This project is a CNN-based image classification model that predicts whether the uploaded image is a cat or a dog.

    ### Technologies Used

    - Python
    - TensorFlow / Keras
    - Streamlit
    

    ### Features

    - Upload image
    - Real-time prediction
    - Confidence score
    - Interactive graph

    ### Dataset

    Trained on a Cats vs Dogs image dataset using Convolutional Neural Networks.

    """)

    st.divider()

    st.success("Made as a mini deep learning project")


# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.markdown(
    """
    <div class='creator'>

    Developed by <b>Piyush Raj</b> | IIT Patna <br>

    Exploring Deep Learning, Computer Vision and AI Projects 🚀

    </div>
    """,
    unsafe_allow_html=True
)