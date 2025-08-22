import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, sawtooth, square
import io
import random
import matplotlib.pyplot as plt

# ------------------ Helpers ------------------
NOTE_FREQS = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46,
    "G5": 783.99, "A5": 880.00, "B5": 987.77
}

SCALES = {
    "C Major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
    "A Minor": ["A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5"],
    "Pentatonic": ["C4", "D4", "E4", "G4", "A4", "C5"]
}

CHORDS = {
    "Cmaj": ["C4", "E4", "G4"], "Fmaj": ["F4", "A4", "C5"], "Gmaj": ["G4", "B4", "D5"],
    "Am": ["A4", "C5", "E5"], "Dm": ["D4", "F4", "A4"], "Em": ["E4", "G4", "B4"]
}

def adsr_envelope(t, sample_rate, attack, decay, sustain_level, release):
    env = np.zeros_like(t)
    attack_end = int(attack * sample_rate)
    decay_end = attack_end + int(decay * sample_rate)
    sustain_end = len(t) - int(release * sample_rate)
    env[:attack_end] = np.linspace(0, 1, attack_end, endpoint=False)
    env[attack_end:decay_end] = np.linspace(1, sustain_level, decay_end - attack_end, endpoint=False)
    env[decay_end:sustain_end] = sustain_level
    env[sustain_end:] = np.linspace(sustain_level, 0, len(t) - sustain_end, endpoint=False)
    return env

def butter_filter(data, cutoff, fs, filter_type="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, data)

def generate_waveform(freq, t, wave_type="Sine"):
    if wave_type == "Sine": return np.sin(2 * np.pi * freq * t)
    if wave_type == "Square": return square(2 * np.pi * freq * t)
    if wave_type == "Saw": return sawtooth(2 * np.pi * freq * t)
    if wave_type == "Triangle": return sawtooth(2 * np.pi * freq * t, width=0.5)
    return np.sin(2 * np.pi * freq * t)

def generate_note(note_name, t, wave_type, adsr, sample_rate):
    freq = NOTE_FREQS.get(note_name, 440.0)
    wave = generate_waveform(freq, t, wave_type)
    env = adsr_envelope(t, sample_rate, *adsr)
    return wave * env

def generate_audio(note_names, duration, wave_type, adsr,
                   filter_on, cutoff, filter_type,
                   sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)
    for note_name in note_names:
        wave += generate_note(note_name, t, wave_type, adsr, sample_rate)
    if filter_on:
        wave = butter_filter(wave, cutoff, sample_rate, filter_type)
    wave = wave / np.max(np.abs(wave))
    audio_int16 = np.int16(wave * 32767)
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer, wave, t

# ------------------ Streamlit UI ------------------
st.title("🎹 Python Music Lab - Mini Synth & Generator")

# ADSR controls
st.subheader("ADSR Envelope")
attack = st.slider("Attack (s)", 0.01, 2.0, 0.2, 0.01)
decay = st.slider("Decay (s)", 0.01, 2.0, 0.3, 0.01)
sustain = st.slider("Sustain Level", 0.0, 1.0, 0.5, 0.01)
release = st.slider("Release (s)", 0.01, 2.0, 0.5, 0.01)

# Waveform + duration
st.subheader("Waveform & Duration")
wave_type = st.selectbox("Waveform", ["Sine", "Square", "Saw", "Triangle"])
duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, 0.1)

# Filter
st.subheader("Filter")
filter_on = st.checkbox("Enable Filter", value=False)
if filter_on:
    filter_type = st.radio("Filter Type", ["low", "high"])
    cutoff = st.slider("Cutoff Frequency (Hz)", 100, 5000, 1000)
else:
    filter_type, cutoff = "low", 1000

# ------------------ Play Options ------------------
st.subheader("Play Options")

col1, col2 = st.columns(2)
with col1:
    note_choice = st.selectbox("🎵 Play a Note", list(NOTE_FREQS.keys()))
    if st.button("Play Note"):
        wav_file, wave, t = generate_audio([note_choice], duration, wave_type,
                                           (attack, decay, sustain, release),
                                           filter_on, cutoff, filter_type)
        st.audio(wav_file, format="audio/wav")
        fig, ax = plt.subplots(2, 1, figsize=(6, 4))
        ax[0].plot(t[:1000], wave[:1000]); ax[0].set_title("Waveform (zoomed)")
        spectrum = np.abs(np.fft.rfft(wave))
        freqs = np.fft.rfftfreq(len(wave), 1/44100)
        ax[1].plot(freqs, spectrum); ax[1].set_title("Spectrum")
        st.pyplot(fig)

with col2:
    chord_choice = st.selectbox("🎶 Play a Chord", list(CHORDS.keys()))
    if st.button("Play Chord"):
        wav_file, wave, t = generate_audio(CHORDS[chord_choice], duration, wave_type,
                                           (attack, decay, sustain, release),
                                           filter_on, cutoff, filter_type)
        st.audio(wav_file, format="audio/wav")

# ------------------ Random Generators ------------------
st.subheader("Random Music Generators")

if st.button("Generate Random Melody"):
    scale = random.choice(list(SCALES.keys()))
    notes = random.choices(SCALES[scale], k=5)
    wav_file, wave, t = generate_audio(notes, duration, wave_type,
                                       (attack, decay, sustain, release),
                                       filter_on, cutoff, filter_type)
    st.write(f"Scale used: {scale}, Notes: {notes}")
    st.audio(wav_file, format="audio/wav")

if st.button("Generate Random Chord Progression"):
    chords = random.choices(list(CHORDS.keys()), k=4)
    all_notes = [note for ch in chords for note in CHORDS[ch]]
    wav_file, wave, t = generate_audio(all_notes, duration, wave_type,
                                       (attack, decay, sustain, release),
                                       filter_on, cutoff, filter_type)
    st.write(f"Chord progression: {chords}")
    st.audio(wav_file, format="audio/wav")

if st.button("Generate Random Sequence (Chords + Notes)"):
    sequence = []
    for _ in range(4):
        if random.random() > 0.5:
            sequence.extend(random.choices(list(CHORDS.keys()), k=1))
        else:
            sequence.extend(random.choices(list(NOTE_FREQS.keys()), k=1))
    all_notes = []
    for s in sequence:
        if s in CHORDS: all_notes.extend(CHORDS[s])
        elif s in NOTE_FREQS: all_notes.append(s)
    wav_file, wave, t = generate_audio(all_notes, duration, wave_type,
                                       (attack, decay, sustain, release),
                                       filter_on, cutoff, filter_type)
    st.write(f"Sequence: {sequence}")
    st.audio(wav_file, format="audio/wav")

# ------------------ Virtual Keyboard ------------------
st.subheader("🎹 Virtual Keyboard")
cols = st.columns(len(NOTE_FREQS))
for i, note_name in enumerate(NOTE_FREQS.keys()):
    if cols[i].button(note_name):
        wav_file, wave, t = generate_audio([note_name], duration, wave_type,
                                           (attack, decay, sustain, release),
                                           filter_on, cutoff, filter_type)
        st.audio(wav_file, format="audio/wav")

# ------------------ Download ------------------
if st.button("Download Last Sound"):
    st.download_button("⬇️ Download WAV", wav_file, file_name="synth_output.wav")
