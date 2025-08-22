import streamlit as st import numpy as np import matplotlib.pyplot as plt import io, wave

--- ADSR Envelope ---

def adsr_envelope(attack, decay, sustain, release, duration, sample_rate=44100): total_len = int(sample_rate * duration) attack_len = int(sample_rate * attack) decay_len = int(sample_rate * decay) release_len = int(sample_rate * release) sustain_len = max(0, total_len - (attack_len + decay_len + release_len))

attack_env = np.linspace(0, 1, attack_len, endpoint=False)
decay_env = np.linspace(1, sustain, decay_len, endpoint=False)
sustain_env = np.full(sustain_len, sustain)
release_env = np.linspace(sustain, 0, release_len, endpoint=True)

env = np.concatenate([attack_env, decay_env, sustain_env, release_env])
env = env[:total_len]  # ensure exact match
return env

--- Tone Generator ---

def generate_tone(freq, duration, attack, decay, sustain, release, sample_rate=44100): t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False) tone = np.sin(2 * np.pi * freq * t) env = adsr_envelope(attack, decay, sustain, release, duration, sample_rate) env = np.pad(env, (0, max(0, len(t)-len(env))), 'constant')[:len(t)] return tone * env

--- Save to WAV ---

def save_to_wav(y, sample_rate=44100): buf = io.BytesIO() with wave.open(buf, 'wb') as wf: wf.setnchannels(1) wf.setsampwidth(2) wf.setframerate(sample_rate) y_int16 = np.int16(y / np.max(np.abs(y)) * 32767) wf.writeframes(y_int16.tobytes()) buf.seek(0) return buf

--- Chord Generator ---

def generate_chord(root_freq, chord_type, duration, attack, decay, sustain, release, sample_rate=44100): if chord_type == "major": freqs = [root_freq, root_freq * (5/4), root_freq * (3/2)] elif chord_type == "minor": freqs = [root_freq, root_freq * (6/5), root_freq * (3/2)] else: freqs = [root_freq]

chord = sum(generate_tone(f, duration, attack, decay, sustain, release, sample_rate) for f in freqs)
return chord / len(freqs)

--- Melody Generator ---

def generate_melody(n_notes, attack, decay, sustain, release, sample_rate=44100): scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C major scale melody = np.array([]) for _ in range(n_notes): freq = np.random.choice(scale) dur = np.random.choice([0.25, 0.5, 1.0]) tone = generate_tone(freq, dur, attack, decay, sustain, release, sample_rate) melody = np.concatenate([melody, tone]) return melody

--- Streamlit UI ---

st.title("🎹 Mini Synthesizer")

st.sidebar.header("ADSR Envelope") attack = st.sidebar.slider("Attack", 0.01, 1.0, 0.1) decay = st.sidebar.slider("Decay", 0.01, 1.0, 0.1) sustain = st.sidebar.slider("Sustain", 0.0, 1.0, 0.7) release = st.sidebar.slider("Release", 0.01, 1.0, 0.2) duration = st.sidebar.slider("Note Duration", 0.1, 2.0, 1.0)

choice = st.radio("Choose what to generate:", ["Single Note", "Chord", "Melody"])

if choice == "Single Note": freq = st.slider("Frequency (Hz)", 100, 1000, 440) if st.button("▶ Play Note"): audio = generate_tone(freq, duration, attack, decay, sustain, release) buf = save_to_wav(audio) st.audio(buf, format="audio/wav")

fig, ax = plt.subplots()
    ax.plot(audio[:2000])
    ax.set_title("Waveform")
    st.pyplot(fig)

elif choice == "Chord": root = st.slider("Root Frequency (Hz)", 100, 600, 261) chord_type = st.selectbox("Chord Type", ["major", "minor"]) if st.button("▶ Play Chord"): audio = generate_chord(root, chord_type, duration, attack, decay, sustain, release) buf = save_to_wav(audio) st.audio(buf, format="audio/wav")

fig, ax = plt.subplots()
    ax.plot(audio[:2000])
    ax.set_title(f"{chord_type.capitalize()} Chord Waveform")
    st.pyplot(fig)

elif choice == "Melody": n_notes = st.slider("Number of Notes", 2, 16, 8) if st.button("▶ Play Melody"): audio = generate_melody(n_notes, attack, decay, sustain, release) buf = save_to_wav(audio) st.audio(buf, format="audio/wav")

fig, ax = plt.subplots()
    ax.plot(audio[:4000])
    ax.set_title("Melody Waveform")
    st.pyplot(fig)

