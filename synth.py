import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import soundfile as sf

def adsr_envelope(attack, decay, sustain, release, duration, sample_rate=44100):
    """Generate ADSR envelope that matches total duration exactly."""
    n_samples = int(duration * sample_rate)

    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)
    s = max(0, n_samples - (a + d + r))  # sustain fills the gap

    attack_curve = np.linspace(0, 1, a, endpoint=False) if a > 0 else np.array([])
    decay_curve = np.linspace(1, sustain, d, endpoint=False) if d > 0 else np.array([])
    sustain_curve = np.full(s, sustain) if s > 0 else np.array([])
    release_curve = np.linspace(sustain, 0, r, endpoint=True) if r > 0 else np.array([])

    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])

    # Ensure exact match
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)), 'constant')
    elif len(env) > n_samples:
        env = env[:n_samples]

    return env

def generate_tone(freq, duration, attack, decay, sustain, release, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    env = adsr_envelope(attack, decay, sustain, release, duration, sample_rate)
    return tone * env

def save_to_wav(y, sample_rate=44100):
    buf = io.BytesIO()
    sf.write(buf, y, sample_rate, format='WAV')
    buf.seek(0)
    return buf

st.title("🎹 Mini Synth with ADSR")

freq = st.slider("Frequency (Hz)", 100, 1000, 440)
duration = st.slider("Duration (s)", 1, 5, 3)

st.subheader("ADSR Envelope")
attack = st.slider("Attack (s)", 0.01, 2.0, 0.1)
decay = st.slider("Decay (s)", 0.01, 2.0, 0.2)
sustain = st.slider("Sustain Level", 0.0, 1.0, 0.7)
release = st.slider("Release (s)", 0.01, 2.0, 0.3)

if st.button("▶ Play Note"):
    audio = generate_tone(freq, duration, attack, decay, sustain, release)
    buf = save_to_wav(audio)
    st.audio(buf, format="audio/wav")

    # Plot waveform
    fig, ax = plt.subplots()
    ax.plot(audio[:2000])  # first few ms
    ax.set_title("Waveform (Zoomed)")
    st.pyplot(fig)

    # Plot ADSR envelope
    env = adsr_envelope(attack, decay, sustain, release, duration)
    fig2, ax2 = plt.subplots()
    ax2.plot(env)
    ax2.set_title("ADSR Envelope")
    st.pyplot(fig2)
