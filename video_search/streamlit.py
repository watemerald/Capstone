import pandas as pd
import streamlit as st

from video_search.models.netvlad import NetVLADModel
from video_search.models.simple_model import SimpleModel
from video_search.utils import predict_url

url = st.text_input("YouTube URL")

model = st.selectbox("Select model", ["NetVLAD", "simple"])

st.write(f"Selected model: {model}")

if url:
    st.video(url)

    if model == "NetVLAD":
        m = NetVLADModel()
    else:
        m = SimpleModel()

    predictions = predict_url(url, m)
    predictions = pd.DataFrame(predictions)
    predictions.columns = ["id", "Category", "Certainty"]

    st.write(predictions.sort_values("Certainty", ascending=False))
