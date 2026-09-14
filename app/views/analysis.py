# MK: Analysis page - the technical detail behind the moods.
# MK: Streamlit turns every file in app/pages/ into its own tab automatically, so this
# MK: shows up next to "Home". Same API, same samples, just more nerdy.

import io
import os

import numpy as np
import requests
import soundfile as sf
import streamlit as st

API_URL = st.secrets["API_URL"]
SAMPLES_DIR = "samples"

st.header("🔬 Analysis – the nerdy stuff")


# MK: ask the API which models are loaded. Reuse Home's answer if it already asked (pages share session_state).
with st.status("Loading models…", expanded=False) as status:
    if "models" not in st.session_state:
        models_info = requests.get(API_URL + "/models", timeout=120).json()
        model_names = []
        for name in models_info["models"]:
            if name != "dummy":
                model_names.append(name)
        if len(model_names) == 0:
            model_names = models_info["models"]
        st.session_state["models"] = model_names
    model_names = st.session_state["models"]
    status.update(label="Models: " + ", ".join(model_names), state="complete", expanded=False)


# MK: 1. pick a sound - a sample here, or reuse whatever is loaded on the Home page
audio_bytes = None
source_name = None

if "audio" in st.session_state:
    if st.checkbox("Use the sound loaded on the Home page", value=True):
        audio_bytes = st.session_state["audio"]
        source_name = st.session_state.get("source_name", "loaded sound")

if audio_bytes is None:
    sample_files = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        if name.endswith(".mp3") or name.endswith(".wav"):
            sample_files.append(name)
    chosen = st.selectbox("Pick a sample", sample_files, index=None, placeholder="Pick a sample…")
    if chosen is not None:
        with open(os.path.join(SAMPLES_DIR, chosen), "rb") as f:
            audio_bytes = f.read()
        source_name = "sample " + chosen

if audio_bytes is None:
    st.info("Pick a sample above (or load a sound on the Home page) to see the analysis.")
    st.stop()

# MK: decode - broken headers can make libsndfile refuse a file, say so instead of crashing
try:
    data, sr = sf.read(io.BytesIO(audio_bytes))
except Exception:
    st.error("Could not read this file. Please use a normal mp3 or wav.")
    st.stop()

if data.ndim > 1:
    data = data.mean(axis=1)        # MK: stereo -> mono
duration = len(data) / sr
st.caption(f"Analysing: {source_name} · {duration:.1f} s · {sr} Hz")


# MK: 2. what the sound looks like: waveform + a spectrogram
st.subheader("What the model hears")

left, right = st.columns([1, 1], gap="large")

with left:
    st.caption("Waveform (loudness over time)")
    step = max(1, len(data) // 2000)
    st.line_chart({"seconds": np.arange(0, len(data), step) / sr, "loudness": np.abs(data[::step])},
                  x="seconds", y="loudness", x_label="seconds", y_label="", height=220)

with right:
    st.caption("Spectrogram (frequency up, time right) – a plain STFT, the same idea as the mel the CNN uses")
    # MK: simple magnitude spectrogram with numpy, so the front end needs no librosa.
    n_fft = 1024
    hop = 512
    window = np.hanning(n_fft)
    if len(data) < n_fft:
        st.info("Too short for a spectrogram.")
    else:
        n_frames = 1 + (len(data) - n_fft) // hop
        frames = []
        i = 0
        while i < n_frames:
            start = i * hop
            frame = data[start:start + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            frames.append(spectrum)
            i = i + 1
        spec = np.array(frames).T                       # MK: shape (frequency bins, time frames)

        # MK: cat sounds sit in the low frequencies. On a high sample-rate file most of the height
        # MK: is near-empty high frequency, so keep only the lowest bins (roughly 0 - 8 kHz).
        max_bin = int(len(spec) * 8000.0 / (sr / 2))
        if max_bin > 1:
            spec = spec[:max_bin]

        spec = 20 * np.log10(spec + 1e-6)               # MK: magnitude -> dB
        # MK: show only the top 50 dB below the loudest point, so the quiet noise floor goes
        # MK: black instead of grey static.
        top = spec.max()
        spec = np.clip(spec, top - 50, top)
        spec = (spec - (top - 50)) / 50                 # MK: normalise that 50 dB window to 0..1
        spec = np.flipud(spec)                          # MK: low frequencies at the bottom
        st.image(spec, use_container_width=True, clamp=True)


# MK: 3. all models on this one sound, side by side
st.subheader("All models on this sound")
st.caption("Same clip, every model. The API takes its own window around the loudest moment.")

verdicts = []
columns = st.columns(len(model_names))
i = 0
while i < len(model_names):
    model = model_names[i]
    with columns[i]:
        st.markdown("**" + model + "**")
        files = {"file": ("clip.mp3", audio_bytes, "audio/mpeg")}
        response = requests.post(API_URL + "/predict", params={"model": model}, files=files, timeout=120)
        if response.status_code != 200:
            st.error("API " + str(response.status_code))
        else:
            result = response.json()
            verdicts.append(result["label"])
            st.metric("verdict", result["label"])
            st.bar_chart(result["probs"], horizontal=True, height=280)
    i = i + 1

# MK: do the models agree? one line, because agreement across architectures is the interesting bit
if len(verdicts) > 0:
    same = True
    for v in verdicts:
        if v != verdicts[0]:
            same = False
    if same:
        st.success("All models agree: " + verdicts[0])
    else:
        st.warning("Models disagree: " + ", ".join(verdicts) + " – this sound sits between classes.")


# MK: 4. the same three models on sounds none of them has seen (fixed numbers from our test run)
st.subheader("On sounds none of the models has ever seen")
st.caption("Four BigSoundBank clips, not in any training set. Label + top probability.")
st.table([
    {"clip": "two cats fighting", "PANNs": "Angry 0.62", "AST": "Angry 0.92", "CNN": "Angry 0.69"},
    {"clip": "cat complaining", "PANNs": "Mating 0.80", "AST": "Warning 0.73", "CNN": "Warning 0.62"},
    {"clip": "cat meow", "PANNs": "Warning 0.29", "AST": "Paining 0.88", "CNN": "Happy 0.47"},
    {"clip": "human imitating a cat", "PANNs": "Happy 1.00", "AST": "Happy 0.59", "CNN": "Happy 0.75"},
])
st.caption("Training clips score ~1.0 on every model – that is memorisation, not accuracy. These unseen clips are the honest picture.")


# MK: 5. how it works, in plain words
st.subheader("How the three models work")
st.markdown(
    "- **PANNs** – a CNN14 pretrained on 2 million audio clips (Google's AudioSet). We freeze it, take its "
    "internal 2048-number summary of a sound, and train a small logistic regression on top to map that to our 10 moods.\n"
    "- **AST** – an Audio Spectrogram Transformer, also pretrained on AudioSet, same trick: freeze it, logistic "
    "regression on top.\n"
    "- **CNN** – a small convolutional net trained from scratch on the cat spectrograms only. No pretraining, so it "
    "starts from nothing but learns the dataset directly.\n"
)

st.subheader("The 'is this even a cat?' check")
st.markdown(
    "PANNs also outputs a probability for each of AudioSet's 527 sound types – Cat, Purr, Meow, Hiss, Caterwaul among "
    "them. We take the highest of those five as a **cat score**. Below 0.2 the API answers *Unknown* instead of a "
    "mood, so speech or noise into the mic doesn't get a fake cat emotion. It does **not** catch a human imitating a "
    "cat (that scores high on 'Meow') – no threshold can, and we say so rather than pretend."
)

st.subheader("What to be careful about")
st.markdown(
    "- **Source leakage** – in the dataset the recording source correlates with the label (some sources only ever "
    "appear for one mood), so a model can partly learn *where a clip came from* instead of the sound itself.\n"
    "- **Duration bias** – clip length alone predicts the mood at 18% (chance is 10%), so we window every clip around "
    "its loudest moment instead of feeding raw length.\n"
    "- **Memorisation** – a model scores ~1.0 on clips it trained on. Only sounds it has never heard tell you anything.\n"
)
