import streamlit as st
import numpy as np
import sounddevice as sd
import io
import random
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# ------------------------------------------
# Utility: Apply ADSR Envelope
# ------------------------------------------
def adsr_envelope(attack, decay, sustain, release, duration, sr=44100):
    attack_samples = int(sr * attack)
    decay_samples = int(sr * decay)
    sustain_samples = int(sr * (duration - attack - decay - release))
    release_samples = int(sr * release)

    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples, endpoint=False), 
        np.linspace(1, sustain, decay_samples, endpoint=False), 
        np.full(sustain_samples, sustain), 
        np.linspace(sustain, 0, release_samples, endpoint=True)
    ])
    return envelope

# ------------------------------------------
# Utility: Low-pass filter
# ------------------------------------------
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# ------------------------------------------
# Synth Note Generator
# ------------------------------------------
def generate_tone(freq=440, duration=1.0, waveform="sine", 
                  attack=0.01, decay=0.1, sustain=0.8, release=0.2,
                  cutoff=1000, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if waveform == "sine":
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "sawtooth":
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform == "triangle":
        wave = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    else:
        wave = np.sin(2 * np.pi * freq * t)

    env = adsr_envelope(attack, decay, sustain, release, duration, sr)
    wave = wave[:len(env)] * env

    # Apply filter
    wave = lowpass_filter(wave, cutoff, sr)
    
    return wave

# ------------------------------------------
# Random melody generator
# ------------------------------------------
def generate_random_melody(duration=10, sr=44100):
    freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C major scale
    melody = np.array([])
    note_duration = 0.8  # longer, slower notes
    while len(melody) < sr * duration:
        freq = random.choice(freqs)
        tone = generate_tone(freq, duration=note_duration, waveform="sine", 
                             attack=0.05, decay=0.2, sustain=0.6, release=0.3, cutoff=800)
        melody = np.concatenate([melody, tone])
    return melody[:int(sr*duration)]

# ------------------------------------------
# Random chords + notes sequence generator
# ------------------------------------------
def generate_random_chords(duration=10, sr=44100):
    chords = [
        [261.63, 329.63, 392.00],   # C major
        [293.66, 369.99, 440.00],   # D minor
        [329.63, 415.30, 493.88],   # E minor
        [349.23, 440.00, 523.25]    # F major
    ]
    sequence = np.array([])
    chord_duration = 1.5
    while len(sequence) < sr * duration:
        chord = random.choice(chords)
        chord_wave = sum([generate_tone(f, duration=chord_duration, waveform="sawtooth",
                                        attack=0.05, decay=0.2, sustain=0.7, release=0.3, cutoff=1200) for f in chord])
        chord_wave /= len(chord)
        sequence = np.concatenate([sequence, chord_wave])
    return sequence[:int(sr*duration)]

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.title("🎹 Python Synth & Melody Generator")

st.sidebar.header("ADSR Controls")
attack = st.sidebar.slider("Attack", 0.01, 1.0, 0.1)
decay = st.sidebar.slider("Decay", 0.01, 1.0, 0.2)
sustain = st.sidebar.slider("Sustain", 0.1, 1.0, 0.7)
release = st.sidebar.slider("Release", 0.01, 1.0, 0.3)

st.sidebar.header("Filter Controls")
cutoff = st.sidebar.slider("Cutoff Frequency (Hz)", 100, 5000, 1000)

waveform = st.selectbox("Waveform", ["sine", "square", "sawtooth", "triangle"])
freq = st.slider("Frequency (Hz)", 100, 1000, 440)
duration = st.slider("Duration (s)", 1, 5, 2)

if st.button("▶️ Play Single Note"):
    wave = generate_tone(freq, duration, waveform, attack, decay, sustain, release, cutoff)
    sd.play(wave, samplerate=44100)
    sd.wait()
    st.success("Played note!")
    st.line_chart(wave[:2000])  # visualize waveform

if st.button("🎵 Generate Random Melody"):
    melody = generate_random_melody()
    sd.play(melody, samplerate=44100)
    sd.wait()
    st.success("Generated peaceful random melody!")
    fig, ax = plt.subplots()
    ax.plot(melody[:5000])
    st.pyplot(fig)

if st.button("🎶 Generate Random Chords + Notes"):
    chords = generate_random_chords()
    sd.play(chords, samplerate=44100)
    sd.wait()
    st.success("Generated random chords + notes sequence!")
    fig, ax = plt.subplots()
    ax.plot(chords[:5000])
    st.pyplot(fig)
