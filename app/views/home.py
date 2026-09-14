# MK: Kitty3000 demo front end. Streamlit, one page, talks to the Cloud Run API.
# MK: The API does all the audio preprocessing (window around the loudest point, per model).
# MK: Layout: left column = input, right column = result, everything fits on one screen.
# MK: Run from the repo root:  uv run streamlit run app/Home.py

import io
import os

import numpy as np
import requests
import soundfile as sf
import streamlit as st

API_URL = st.secrets["API_URL"]
SAMPLES_DIR = "samples"


st.header("🐱 Kitty3000 – what does your cat say?")


# MK: 0. warm-up. The API sleeps when nobody uses it and needs up to a minute to load the models on the
# MK: first call. We do that call once, right at the start, and remember the answer in the session.
with st.status("Waking up the cat… loading models, this can take up to a minute", expanded=False) as status:
    if "models" not in st.session_state:
        st.write("Asking the API which models are loaded")
        models_info = requests.get(API_URL + "/models", timeout=120).json()
        model_names = []
        for name in models_info["models"]:
            if name != "dummy":
                model_names.append(name)
        if len(model_names) == 0:
            model_names = models_info["models"]     # MK: only the dummy is loaded, show it anyway
        st.session_state["models"] = model_names
    model_names = st.session_state["models"]
    status.update(label="Models ready: " + ", ".join(model_names), state="complete", expanded=False)

left, right = st.columns([1, 1], gap="large")


# MK: 1. left column: pick ONE source (sample, recording or upload), then the model
with left:
    # MK: two states: nothing loaded -> show the three tabs; something loaded -> show it with a Clear button.
    # MK: The loaded sound is kept in st.session_state so it survives every rerun (slider, model choice, Analyse).
    if "audio" not in st.session_state:
        tab_upload, tab_record, tab_samples = st.tabs(["Upload", "Record", "Samples"])

        with tab_upload:
            uploaded = st.file_uploader("mp3 or wav", type=["mp3", "wav"], label_visibility="collapsed")
            if uploaded is not None:
                st.session_state["audio"] = uploaded.getvalue()
                st.session_state["source_name"] = "file " + uploaded.name
                st.rerun()

        with tab_record:
            recorded = st.audio_input("Record your cat", label_visibility="collapsed")
            if recorded is not None:
                st.session_state["audio"] = recorded.getvalue()
                st.session_state["source_name"] = "your recording"
                st.rerun()

        with tab_samples:
            sample_files = []
            for name in sorted(os.listdir(SAMPLES_DIR)):
                if name.endswith(".mp3") or name.endswith(".wav"):
                    sample_files.append(name)
            chosen = st.selectbox("Pick a sample", sample_files, index=None, placeholder="Pick a sample…", label_visibility="collapsed")
            if chosen is not None:
                with open(os.path.join(SAMPLES_DIR, chosen), "rb") as f:
                    st.session_state["audio"] = f.read()
                st.session_state["source_name"] = "sample " + chosen
                st.rerun()

        st.info("Upload a file, record your cat or pick a sample to start.")
        st.stop()

    audio_bytes = st.session_state["audio"]
    source_name = st.session_state["source_name"]

    if st.button("✕ Clear and load another sound"):
        del st.session_state["audio"]
        del st.session_state["source_name"]
        st.rerun()

    # MK: show what is loaded and let the user (optionally) pick a part of it.
    # MK: Some mp3s have broken headers (phones, sound archives) and libsndfile refuses them - say so instead of crashing.
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        st.error("Could not read this file. Please use a normal mp3 or wav.")
        st.stop()

    if data.ndim > 1:
        data = data.mean(axis=1)        # MK: stereo -> mono, only for the waveform and the trimming
    duration = len(data) / sr

    st.caption(f"Loaded: {source_name} · {duration:.1f} s")
    st.audio(audio_bytes)

    step = max(1, len(data) // 2000)    # MK: waveform with ~2000 points, so the chart stays light
    seconds = np.arange(0, len(data), step) / sr
    st.line_chart({"seconds": seconds, "loudness": np.abs(data[::step])}, x="seconds", y="loudness",
                  x_label="seconds", y_label="", height=110)

    with st.expander("Only analyse a part of the recording"):
        # MK: range slider: drag a dot to resize the snippet, click the track to jump the nearest dot there.
        start, end = st.slider("Seconds", 0.0, float(duration), (0.0, float(duration)), 0.1)
        snippet = data[int(start * sr):int(end * sr)]

        # MK: the API takes its own window around the loudest moment (11 s for PANNs / AST, 4 s chunks for the CNN),
        # MK: so the user does not need to cut precisely. We send the snippet as mp3.
        buffer = io.BytesIO()
        sf.write(buffer, snippet, sr, format="MP3")
        snippet_bytes = buffer.getvalue()

        # MK: second player, so the user hears exactly the part that goes to the model
        st.caption(f"This part goes to the model: {start:.1f} – {end:.1f} s ({end - start:.1f} s)")
        st.audio(snippet_bytes)

    #model = st.radio("Which model should listen?", model_names, horizontal=True)
    cat_icons = {"panns": "🦁 PANNs", "ast": "🤖 AST", "cnn": "🐱 CNN"}
    labels = []
    label_to_model = {}
    for name in model_names:
        if name in cat_icons:
            label = cat_icons[name]
        else:
            label = "🐾 " + name        # MK: a new model still gets a button, just a paw print
        labels.append(label)
        label_to_model[label] = name

    chosen_label = st.segmented_control("Which cat should listen?", labels, default=labels[0])
    if chosen_label is None:
        chosen_label = labels[0]        # MK: segmented_control returns None if you deselect, keep a default
    model = label_to_model[chosen_label]
    analyse = st.button("Analyse", type="primary", use_container_width=True)


# MK: 2. right column: the result
with right:
    st.subheader("Mood")

    if not analyse:
        st.caption("Load a sound and a model on the left, then press Analyse.")
        st.stop()

    with st.spinner(f"{model} is listening…"):
        files = {"file": ("snippet.mp3", snippet_bytes, "audio/mpeg")}
        response = requests.post(API_URL + "/predict", params={"model": model}, files=files, timeout=120)

    if response.status_code != 200:
        st.error(f"The API answered {response.status_code}: {response.text}")
        st.stop()

    result = response.json()
    label = result["label"]
    probs = result["probs"]

    # MK: the result: label big, the ten probabilities as bars.
    # MK: If the API did not hear a cat, label is "Unknown" and not in probs, so we show the cat score instead.
    if result["is_cat"]:
        st.metric("Your cat is", label, f"{probs[label]:.0%} confident · model: {model}", delta_color="off")
    else:
        st.metric("Your cat is", "not a cat?", f"cat score {result['cat_score']:.2f} · model: {model}", delta_color="off")
        st.warning("This does not sound like a cat. The mood below is what the model would guess anyway.")
    st.bar_chart(probs, horizontal=True, height=330)
