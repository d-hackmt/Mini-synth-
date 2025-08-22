import streamlit as st
import numpy as np
import sounddevice as sd
import io
import random
import matplotlib.pyplot as plt

# ---------------------------------------
# Synth Core
# ---------------------------------------

SAMPLE_RATE = 44100

def adsr_envelope(attack, decay, sustain, release, duration):
    attack_samples = int(attack * SAMPLE_RATE)
    decay_samples = int(decay * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)
    sustain_samples = int(duration * SAMPLE_RATE) - (attack_samples + decay_samples + release_samples)

    if sustain_samples < 0:
        sustain_samples = 0

    attack_env = np.linspace(0, 1, attack_samples, False)
    decay_env = np.linspace(1, sustain, decay_samples, False)
    sustain_env = np.ones(sustain_samples) * sustain
    release_env = np.linspace(sustain, 0, release_samples, False)

    return np.concatenate((attack_env, decay_env, sustain_env, release_env))


def generate_wave(freq, duration, adsr, waveform="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    if waveform == "sine":
        wave = np.sin(freq * t * 2 * np.pi)
    elif waveform == "square":
        wave = np.sign(np.sin(freq * t * 2 * np.pi))
    elif waveform == "saw":
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform == "triangle":
        wave = 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
    else:
        wave = np.sin(freq * t * 2 * np.pi)

    env = adsr_envelope(*adsr, duration)
    wave = wave[:len(env)] * env
    return wave


def play_sound(wave):
    sd.play(wave, SAMPLE_RATE)
    sd.wait()


# ---------------------------------------
# Music Generation Functions
# ---------------------------------------

NOTES = {"C":261.63,"D":293.66,"E":329.63,"F":349.23,"G":392.00,"A":440.00,"B":493.88}
CHORDS = {
    "Cmaj":[261.63,329.63,392.00],
    "Fmaj":[349.23,440.00,523.25],
    "Gmaj":[392.00,493.88,587.33],
    "Amin":[440.00,523.25,659.25]
}

def random_melody(adsr, waveform, duration=10):
    melody = np.array([])
    note_duration = duration / 8  # 8 notes spread
    for _ in range(8):
        note = random.choice(list(NOTES.values()))
        melody = np.concatenate((melody, generate_wave(note, note_duration, adsr, waveform)))
    return melody


def random_chord_progression(adsr, waveform, duration=10):
    progression = np.array([])
    chord_duration = duration / 4  # 4 chords
    for _ in range(4):
        chord = random.choice(list(CHORDS.values()))
        chord_wave = sum(generate_wave(n, chord_duration, adsr, waveform) for n in chord) / len(chord)
        progression = np.concatenate((progression, chord_wave))
    return progression


def random_sequence(adsr, waveform, duration=10):
    seq = np.array([])
    section_duration = duration / 6
    for _ in range(6):
        if random.random() > 0.5:  # sometimes play note
            note = random.choice(list(NOTES.values()))
            seq = np.concatenate((seq, generate_wave(note, section_duration, adsr, waveform)))
        else:  # sometimes play chord
            chord = random.choice(list(CHORDS.values()))
            chord_wave = sum(generate_wave(n, section_duration, adsr, waveform) for n in chord) / len(chord)
            seq = np.concatenate((seq, chord_wave))
    return seq


# ---------------------------------------
# Streamlit UI
# ---------------------------------------

st.title("🎹 Mini Synth with ADSR, Chords & Melodies")

A = st.slider("Attack", 0.01, 2.0, 0.1)
D = st.slider("Decay", 0.01, 2.0, 0.2)
S = st.slider("Sustain", 0.0, 1.0, 0.7)
R = st.slider("Release", 0.01, 2.0, 0.5)
waveform = st.selectbox("Waveform", ["sine", "square", "saw", "triangle"])

adsr = (A, D, S, R)

if st.button("🎶 Play Random Melody (10s)"):
    wave = random_melody(adsr, waveform, duration=10)
    play_sound(wave)

if st.button("🎵 Play Random Chord Progression (10s)"):
    wave = random_chord_progression(adsr, waveform, duration=10)
    play_sound(wave)

if st.button("🎼 Play Random Sequence (Notes + Chords, 10s)"):
    wave = random_sequence(adsr, waveform, duration=10)
    play_sound(wave)

# Waveform visualization
if st.button("📊 Visualize Last Waveform"):
    if 'wave' in locals():
        fig, ax = plt.subplots()
        ax.plot(wave[:2000])
        st.pyplot(fig)
