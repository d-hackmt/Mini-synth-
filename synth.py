import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import io
import matplotlib.pyplot as plt

# --- Utility functions ---

def adsr_envelope(attack, decay, sustain, release, duration, sample_rate=44100):
    """Generate ADSR envelope"""
    a = int(sample_rate * attack)
    d = int(sample_rate * decay)
    s = int(sample_rate * (duration - (attack + decay + release)))
    r = int(sample_rate * release)

    attack_curve = np.linspace(0, 1, a, endpoint=False)
    decay_curve = np.linspace(1, sustain, d, endpoint=False)
    sustain_curve = np.ones(max(s,0)) * sustain
    release_curve = np.linspace(sustain, 0, max(r,0), endpoint=False)

    return np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])

def generate_tone(freq, duration, attack, decay, sustain, release, sample_rate=44100):
    """Generate a tone with ADSR envelope"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * freq * t)
    env = adsr_envelope(attack, decay, sustain, release, duration, sample_rate)
    env = np.pad(env, (0, max(0, len(t)-len(env))), 'constant')
    return tone * env

def save_to_wav(y, sample_rate=44100):
    """Save numpy array to wav in memory"""
    y = np.int16(y/np.max(np.abs(y)) * 32767)
    buf = io.BytesIO()
    write(buf, sample_rate, y)
    buf.seek(0)
    return buf

# --- Streamlit UI ---
st.title("🎹 Streamlit Synth (Browser Compatible)")

# ADSR Sliders
attack = st.slider("Attack", 0.01, 2.0, 0.1, 0.01)
decay = st.slider("Decay", 0.01, 2.0, 0.2, 0.01)
sustain = st.slider("Sustain", 0.0, 1.0, 0.7, 0.05)
release = st.slider("Release", 0.01, 3.0, 0.5, 0.01)

duration = st.slider("Duration (sec)", 1, 10, 5)

# Frequency / Note choice
freq = st.slider("Frequency (Hz)", 100, 1000, 440)

if st.button("▶ Play Note"):
    audio = generate_tone(freq, duration, attack, decay, sustain, release)
    buf = save_to_wav(audio)
    st.audio(buf, format="audio/wav")

# Random Melodies
if st.button("🎶 Generate Random Melody"):
    freqs = np.random.choice([220, 330, 440, 550, 660, 770], size=8)
    song = []
    for f in freqs:
        note = generate_tone(f, duration/len(freqs), attack, decay, sustain, release)
        song.append(note)
    melody = np.concatenate(song)
    buf = save_to_wav(melody)
    st.audio(buf, format="audio/wav")

# --- ADSR Visualizer ---
st.subheader("ADSR Envelope Visualizer")
env = adsr_envelope(attack, decay, sustain, release, duration)
fig, ax = plt.subplots()
ax.plot(env, color="purple")
ax.set_title("ADSR Shape")
st.pyplot(fig)
