import streamlit as st
import numpy as np
import sounddevice as sd
import io
import random
import matplotlib.pyplot as plt

# Sampling rate
SAMPLE_RATE = 44100  

# ADSR Envelope
def adsr_envelope(duration, attack, decay, sustain, release, sustain_level=0.7):
    total_samples = int(SAMPLE_RATE * duration)
    attack_samples = int(attack * SAMPLE_RATE)
    decay_samples = int(decay * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)
    sustain_samples = max(0, total_samples - (attack_samples + decay_samples + release_samples))

    # Attack
    attack_curve = np.linspace(0, 1, attack_samples)
    # Decay
    decay_curve = np.linspace(1, sustain_level, decay_samples)
    # Sustain
    sustain_curve = np.ones(sustain_samples) * sustain_level
    # Release
    release_curve = np.linspace(sustain_level, 0, release_samples)

    envelope = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])
    if len(envelope) < total_samples:
        envelope = np.pad(envelope, (0, total_samples - len(envelope)), 'constant')
    return envelope

# Oscillator (sine wave)
def osc(frequency, duration, attack, decay, sustain, release, waveform="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    if waveform == "sine":
        wave = np.sin(2 * np.pi * frequency * t)
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "saw":
        wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    else:
        wave = np.sin(2 * np.pi * frequency * t)

    envelope = adsr_envelope(duration, attack, decay, sustain, release)
    return wave * envelope

# Play sound
def play_sound(audio):
    sd.stop()
    sd.play(audio, samplerate=SAMPLE_RATE)

# Random melody generator
def generate_random_melody(duration, attack, decay, sustain, release):
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C major scale
    note_duration = 0.5
    melody = np.array([])

    for _ in range(int(duration / note_duration)):
        f = random.choice(freqs)
        note = osc(f, note_duration, attack, decay, sustain, release, waveform="sine")
        melody = np.concatenate([melody, note])
    return melody

# Random chord progression
def generate_random_chords(duration, attack, decay, sustain, release):
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [293.66, 349.23, 440.00],  # D minor
        [329.63, 392.00, 493.88],  # E minor
        [349.23, 440.00, 523.25],  # F major
    ]
    chord_duration = 1.0
    sequence = np.array([])

    for _ in range(int(duration / chord_duration)):
        chord = random.choice(chords)
        layered_chord = sum([osc(f, chord_duration, attack, decay, sustain, release) for f in chord]) / len(chord)
        sequence = np.concatenate([sequence, layered_chord])
    return sequence

# Combined sequence (melody + chords layered)
def generate_random_sequence(duration, attack, decay, sustain, release):
    melody = generate_random_melody(duration, attack, decay, sustain, release)
    chords = generate_random_chords(duration, attack, decay, sustain, release)
    min_len = min(len(melody), len(chords))
    return (melody[:min_len] + chords[:min_len]) / 2.0

# Streamlit UI
st.title("🎹 ADSR Synth with Random Music Generator")

# Sliders for ADSR
attack = st.slider("Attack", 0.01, 2.0, 0.1, 0.01)
decay = st.slider("Decay", 0.01, 2.0, 0.2, 0.01)
sustain = st.slider("Sustain", 0.01, 5.0, 0.5, 0.01)
release = st.slider("Release", 0.01, 3.0, 0.5, 0.01)

# Buttons
if st.button("🎵 Play Random Melody"):
    melody = generate_random_melody(10, attack, decay, sustain, release)
    play_sound(melody)

if st.button("🎶 Play Random Chords"):
    chords = generate_random_chords(10, attack, decay, sustain, release)
    play_sound(chords)

if st.button("🎼 Play Combined Sequence (Melody + Chords)"):
    seq = generate_random_sequence(10, attack, decay, sustain, release)
    play_sound(seq)

# ADSR Visualizer
if st.button("📊 Show ADSR Envelope"):
    env = adsr_envelope(2, attack, decay, sustain, release)
    plt.figure(figsize=(8,4))
    plt.plot(env)
    plt.title("ADSR Envelope Shape")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)
