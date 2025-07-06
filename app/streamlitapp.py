# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
import numpy as np
import cv2
import time

from utils import load_data, num_to_char
from modelutil import load_model

from streamlit_extras.let_it_rain import rain
from streamlit_extras.metric_cards import style_metric_cards

# Set the layout
st.set_page_config(layout='wide', page_title="LipBuddy")

# Show full-screen landing page only once
if "show_app" not in st.session_state:
    st.session_state["show_app"] = False

# Function: Landing page with full-screen image + centered button
def show_landing():
    # Fullscreen image as background
    st.markdown(
        """
        <style>
        .landing-image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            height: 7vh;
            background-color: black;
        }

        .landing-img {
            max-height: 50vh;
            width: 100%;
            object-fit: cover;
        }

        .button-container {
            margin-top: -1000px;  /* move button upwards over the image */
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # Create layout using columns to center button
    st.markdown("<div class='landing-image-container'>", unsafe_allow_html=True)
    st.image("static/landing.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Centered button using Streamlit layout
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state["show_app"] = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)











# Function: Main App
def main_app():
    with st.sidebar:
        st.image(
            'https://img.izismile.com/img/img4/20110201/1000/stunning_animated_gifs_22.gif',
            use_container_width=True
        )

        st.markdown("""
            <h2 style='text-align: center; color: #6C63FF; margin-top: 10px;'>🎥 Whispr.AI</h2>
            <p style='text-align: center; color: #999;'>Lip Reading with Deep Learning</p>
            <hr style='border: 1px solid #EEE'>
        """, unsafe_allow_html=True)

        st.subheader("🗣️User Info")
        username = st.text_input("Your Name", key="user_name")
        model_choice = st.radio("Choose Model", ["LipNet"], key="model_choice")

        if username:
            st.success(f"👋 Welcome, {username}!")
        else:
            st.info("Enter your name to personalize experience.")

        with st.expander("ℹ️ About This App", expanded=False):
            st.markdown("""
                **Whispr.AI** is powered by the original **LipNet** deep learning model, trained for video-to-text lip reading.
                - 🔬 Technologies: TensorFlow, Conv3D, BiLSTM, CTC
                - 🎯 Input: Video clip of lip movement
                - 📝 Output: Predicted sentence
            """)
        
        

        # --- System Stats ---
        st.markdown("### 📈 System Stats", unsafe_allow_html=True)

        stats_html = """
        <style>
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px 16px;
            font-size: 13px;
            margin-top: 10px;
        }

        .stat-box {
            background-color: #f0f2f6;
            padding: 3px;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            color: #333;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stat-box:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>

        <div class="stats-grid">
            <div class="stat-box">🎥 <strong>FPS:</strong><br>25</div>
            <div class="stat-box">🧮 <strong>Frames:</strong><br>75</div>
            <div class="stat-box">🖼️ <strong>Resolution:</strong><br>140x46</div>
            <div class="stat-box">🤖 <strong>Model:</strong><br>LipNet</div>
        </div>
        """

        st.markdown(stats_html, unsafe_allow_html=True)
        st.write(" ")

       

        
        


        # Optional: Inference time, if calculated later
    

        



        # rain(
        #     emoji="🧠",
        #     font_size=18,
        #     falling_speed=5,
        #     animation_length="infinite"
        # )
        # --- Theme Switch ---
        theme_choice = st.radio("🎨 Choose Theme", ["Dark", "Light"], key="theme_mode")

        if theme_choice == "Dark":
                st.markdown(
                """
                <style>
                body {
                    background-color: #1e1e1e;
                    color: white;
                }
                .stApp {
                    background-color: #1e1e1e;
                    color: white;
                }
                </style>
                """,
                    unsafe_allow_html=True
            )
        else:
                st.markdown(
                """
                <style>
                    body {
                        background-color: #ffffff;
                        color: black;
                    }
                    .stApp {
                        background-color: #ffffff;
                        color: black;
                    }
                </style>
                """,
                    unsafe_allow_html=True
                )


        st.markdown("""
            <hr style='border: 0.5px solid #DDD'>
            <div style='text-align: center; font-size: 12px; color: grey;'>
                Made with ❤️ by Madan<br>
                <a href='https://github.com/Madan-A' target='_blank'>GitHub</a> • 
                <a href='https://www.linkedin.com/in/madan15/'>LinkedIn</a>
            </div>
        """, unsafe_allow_html=True)

    


    # --- HEADER + WELCOME SECTION ---
    user = st.session_state.get("user_name", "Guest")

    st.markdown("""
        <style>
        .header-container {
            text-align: center;
            padding-top: 30px;
            padding-bottom: 20px;
        }
        .main-title {
            font-size: 42px;
            color: #6C63FF;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .welcome-text {
            font-size: 20px;
            color: white;
            margin-bottom: 12px;
        }
        .description {
            font-size: 16px;
            color: white;
            max-width: 750px;
            margin: 0 auto;
            line-height: 1.6;
            text-align:left;
                
        }
        </style>
        <div class="header-container">
            <div class="main-title">👄 Welcome to Whisper.AI</div>
            <div class="welcome-text">Hey get ready to experience real-time lip reading powered by AI!</div>
            <div class="description">
                <strong>Whisper.AI</strong> is an AI-powered application built on top of the <strong>LipNet</strong> deep learning model.
                It reads video clips of lip movements and predicts spoken sentences using advanced computer vision and sequence modeling techniques.
                <br>
                This project showcases the power of Conv3D,BiLSTM, and CTC decoding to understand human speech visually — just like how deaf individuals read lips.
            </div>
        </div>
    """, unsafe_allow_html=True)



    # --- HOW IT WORKS SECTION ---
    st.markdown("<h3 style='color:#6C63FF'>🔍 How It Works:</h3>", unsafe_allow_html=True)

    with st.expander("📥 1. Input Preprocessing"):
        st.markdown("""
        - The input is a `.mpg` video (25 FPS, 75 frames per clip).
        - Each frame is resized to **(46x140)** and converted to grayscale.
        - The entire video is normalized and shaped into a tensor: **(75, 46, 140, 1)**.
        - Tensor format: *(frames, height, width, channels)*.
        """)

    with st.expander("🧠 2. Model Architecture"):
        st.markdown("""
        The backbone model is **LipNet**, which follows:
        - `Conv3D` layers to extract spatio-temporal features.
        - `MaxPool3D` layers to reduce dimensionality.
        - `TimeDistributed` and `BiLSTM` layers to learn temporal dependencies.
        - `Dense` + `softmax` to classify each time step.
        
        **Output Shape:** (Time steps, Vocabulary Size = 41)
        """)

    with st.expander("🧬 3. Connectionist Temporal Classification (CTC) Decoding"):
        st.markdown("""
        - Since we don’t know exact alignments between frames and text, we use **CTC Loss**.
        - It allows the model to learn alignment-free sequence-to-sequence mapping.
        - Decoding is done using greedy search: `tf.keras.backend.ctc_decode()`.
        """)

    with st.expander("📝 4. Final Output Prediction"):
        st.markdown("""
        - The predicted output is a sequence of token IDs (integers).
        - These tokens are mapped to characters using a `num_to_char` lookup.
        - Final sentence is reconstructed using `tf.strings.reduce_join(...)`.
        """)

    with st.expander("🎨 5. Post-processing & Visualization"):
        st.markdown("""
        - We render the original `.mp4` video for reference.
        - The model’s input (preprocessed frames) is converted into an **animated GIF**.
        - You see token predictions and final decoded sentence in real-time.
        """)




    options = os.listdir(os.path.join('..', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)

    col1, col2 = st.columns(2)

    if options:
        with col1:
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('..', 'data', 's1', selected_video)

            os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 test_video.mp4 -y')

            with open('test_video.mp4', 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes, format="video/mp4")
            st.write(" ")
            
            
            st.markdown("## 📊 Prediction Dashboard")
            st.write(" ")


        with col2:
            st.info('This is all the machine learning model sees when making a prediction')

            video, annotations = load_data(tf.convert_to_tensor(file_path))
            video = video.numpy() if isinstance(video, tf.Tensor) else video
            if video.shape[-1] == 1:
                video = np.squeeze(video, axis=-1)

            processed_video = []
            for frame in video:
                norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                norm_frame = norm_frame.astype(np.uint8)
                norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2RGB)
                processed_video.append(norm_frame)

            imageio.mimsave('animation.gif', processed_video, fps=10)
            st.image('animation.gif', width=400)

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
            st.write(" ")




            if "accuracy_data" not in st.session_state:
                st.session_state["accuracy_data"] = []

            # Simulate ground truth for now — later you can load actual annotations
            ground_truth_tensor = tf.strings.reduce_join(num_to_char(annotations)).numpy()
            ground_truth = ground_truth_tensor.decode("utf-8")

            prediction = converted_prediction

            # Calculate word-level accuracy (you can also use Levenshtein for char-wise)
            def calculate_accuracy(gt, pred):
                gt_words = gt.split()
                pred_words = pred.split()
                correct = sum(1 for w1, w2 in zip(gt_words, pred_words) if w1 == w2)
                return round(100 * correct / max(len(gt_words), 1), 2)

            acc = calculate_accuracy(ground_truth, prediction)

            # Add to session state
            st.session_state["accuracy_data"].append({
                "video": selected_video,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "accuracy": acc
            })



            import pandas as pd
            import altair as alt

            

        if "accuracy_data" in st.session_state and st.session_state["accuracy_data"]:
            df = pd.DataFrame(st.session_state["accuracy_data"])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🧾 Prediction Summary")
                st.dataframe(df[["video", "ground_truth", "prediction", "accuracy"]], use_container_width=True)

            with col2:
                st.markdown("#### 📈 Accuracy Graph")
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("video:N", title="Video"),
                    y=alt.Y("accuracy:Q", title="Accuracy (%)"),
                    tooltip=["video", "accuracy"]
                ).properties(
                    width=400,
                    height=300,
                    title="Accuracy per Video"
                )
     
                st.altair_chart(chart, use_container_width=True)
    


    st.markdown("### 💬 Feedback")

    user_feedback = st.text_area("How do you feel about this prediction?", placeholder="Write here...")

    if st.button("Submit Feedback"):
        st.success("✅ Feedback submitted! Thank you.")
        # (Optional) Save it to a file or DB





if "show_app" not in st.session_state:
    st.session_state["show_app"] = False

if not st.session_state["show_app"]:
    show_landing()
else:
    main_app()
