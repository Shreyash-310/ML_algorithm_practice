import os
import streamlit as st
import tempfile
import whisper
import google.generativeai as genai
from app_MoM import *


# Streamlit App code
st.title("Audio Transcription and Meeting Minutes")

transcript_generator = GenerateTranscript()
minutes_generator = GenerateMinutes()

# Initialize the session state for transcript and minutes
# if 'transcript' not in st.session_state:
#     st.session_state.transcript = None
# if 'minutes' not in st.session_state:
#     st.session_state.minutes = None

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "m4a", "wav"])

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file, format=uploaded_file.type)
    # print(f"$$$ uploaded_file => {uploaded_file.name}")

    # if st.button("Generate Transcript"):
    if st.button("Generate Minutes of Meeting"):
            transcript = transcript_generator.get_transcribe(uploaded_file.name)
        # st.write(transcript)
        # st.text_area("Transcript", value=transcript, height=300)

        # if st.button("Generate Minutes of Meeting"):
            print("Summarization")
            minutes = minutes_generator.summarize(transcript)
            st.text_area("Minutes of Meeting", value=minutes, height=200)
            
            if st.button("Save MoM"):
                st.write("Saving the MoM file")