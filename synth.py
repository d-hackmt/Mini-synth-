import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, sawtooth, square
import io

# ---- Helper functions ----
def adsr_envelope(t, sample_rate, attack, decay, sustain_level, release):
    env = np.zeros_like(t)
    attack_end = int(attack * sample_rate)
    decay_end = attack_end + int(decay * sample_rate)
    sustain_end = len(t) - int(release * sample_rate)

    # Attack
    env[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)
    # Decay
    env[attack_end:decay_end] = np.linspace(1, sustain_level, decay_end - attack_end, endpoint=False)
    # Sustain
    env[decay_end:sustain_end] = sustain_level
    # Release
    env[sustain_end:] = np.linspace(sustain_level, 0, len(t) - sustain_end, endpoint=False)
    return env

def butter_filter(data, cutoff, fs, filter_type="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, data)

def generate_waveform(freq, t, wave_type="Sine"):
    if wave_type == "Sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square":
        return square(2 * np.pi * freq * t)
    elif wave_type == "Saw":
        return sawtooth(2 * np.pi * freq * t)
    elif wave_type == "Triangle":
        return sawtooth(2 * np.pi * freq * t, width=0.5)
    return np.sin(2 * np.pi * freq * t)

def apply_lfo(signal, t, rate, depth, mode="Tremolo"):
    lfo = np.sin(2 * np.pi * rate * t) * depth
    if mode == "Tremolo":
        return signal * (1 + lfo)  # amplitude modulation
    elif mode == "Vibrato":
        # vibrato = frequency modulation (simple approx)
        modulated = np.sin(2 * np.pi * (440 + 20 * lfo) * t)
        return modulated
    return signal

# ---- Cached Tone Generator ----
@st.cache_data
def generate_tone(attack, decay, sustain, release, freq, duration,
                  wave_type, lfo_rate, lfo_depth, lfo_mode, 
                  filter_on, cutoff, filter_type,
                  sample_rate=44100):

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate waveform
    wave = generate_waveform(freq, t, wave_type)

    # Apply ADSR
    env = adsr_envelope(t, sample_rate, attack, decay, sustain, release)
    wave = wave * env

    # Apply LFO
    if lfo_depth > 0:
        wave = apply_lfo(wave, t, lfo_rate, lfo_depth, lfo_mode)

    # Apply Filter
    if filter_on:
        wave = butter_filter(wave, cutoff, sample_rate, filter_type)

    # Normalize
    wave = wave / np.max(np.abs(wave))

    # Convert to WAV buffer
    audio_int16 = np.int16(wave * 32767)
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer

# ---- Streamlit UI ----
st.title("🎹 Python Mini Synth (Streamlit)")

st.subheader("ADSR Envelope")
attack = st.slider("Attack (s)", 0.01, 2.0, 0.2, 0.01)
decay = st.slider("Decay (s)", 0.01, 2.0, 0.3, 0.01)
sustain = st.slider("Sustain Level", 0.0, 1.0, 0.5, 0.01)
release = st.slider("Release (s)", 0.01, 2.0, 0.5, 0.01)

st.subheader("Waveform")
wave_type = st.selectbox("Waveform", ["Sine", "Square", "Saw", "Triangle"])
freq = st.slider("Frequency (Hz)", 50, 2000, 440)

st.subheader("LFO")
lfo_rate = st.slider("LFO Rate (Hz)", 0.1, 20.0, 5.0, 0.1)
lfo_depth = st.slider("LFO Depth", 0.0, 1.0, 0.0, 0.01)
lfo_mode = st.selectbox("LFO Mode", ["Tremolo", "Vibrato"])

st.subheader("Filter")
filter_on = st.checkbox("Enable Filter", value=False)
if filter_on:
    filter_type = st.radio("Filter Type", ["low", "high"])
    cutoff = st.slider("Cutoff Frequency (Hz)", 100, 5000, 1000)
else:
    filter_type = "low"
    cutoff = 1000

duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, 0.1)

if st.button("Play Sound"):
    wav_file = generate_tone(attack, decay, sustain, release,
                             freq, duration, wave_type,
                             lfo_rate, lfo_depth, lfo_mode,
                             filter_on, cutoff, filter_type)
    st.audio(wav_file, format="audio/wav")
