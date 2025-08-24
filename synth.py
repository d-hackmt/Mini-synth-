import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, sawtooth, square
import io
import matplotlib.pyplot as plt

# =========================
# ---------- DSP ----------
# =========================
SR = 44100

def adsr_envelope(total_samples, attack, decay, sustain_level, release):
    """Return an ADSR envelope of length total_samples."""
    a = int(max(1, attack * SR))
    d = int(max(1, decay * SR))
    r = int(max(1, release * SR))
    s = max(0, total_samples - (a + d + r))
    env = np.zeros(total_samples, dtype=np.float32)
    idx = 0
    # Attack
    if a > 0:
        env[idx:idx+a] = np.linspace(0.0, 1.0, a, endpoint=False)
        idx += a
    # Decay
    if d > 0:
        env[idx:idx+d] = np.linspace(1.0, sustain_level, d, endpoint=False)
        idx += d
    # Sustain
    if s > 0:
        env[idx:idx+s] = sustain_level
        idx += s
    # Release
    if r > 0:
        env[idx:idx+r] = np.linspace(sustain_level, 0.0, r, endpoint=False)
        idx += r
    # Pad if rounding shaved a sample
    if idx < total_samples:
        env[idx:] = 0.0
    return env

def butter_filter(data, cutoff, fs, filter_type="low"):
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 0.999999)
    btype = "lowpass" if filter_type == "low" else "highpass"
    b, a = butter(4, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

def gen_wave(freq, t, wave_type="Sine"):
    if wave_type == "Sine":
        return np.sin(2*np.pi*freq*t, dtype=np.float32)
    if wave_type == "Square":
        return square(2*np.pi*freq*t).astype(np.float32)
    if wave_type == "Saw":
        return sawtooth(2*np.pi*freq*t).astype(np.float32)
    if wave_type == "Triangle":
        return sawtooth(2*np.pi*freq*t, width=0.5).astype(np.float32)
    return np.sin(2*np.pi*freq*t, dtype=np.float32)

def apply_lfo(signal, t, rate, depth, mode="Tremolo", base_freq=None, wave_type="Sine"):
    if depth <= 0:
        return signal
    lfo = np.sin(2*np.pi*rate*t) * depth
    if mode == "Tremolo":
        # amplitude modulation (keep always positive gain)
        return signal * (1.0 + 0.5*lfo).astype(np.float32)
    elif mode == "Vibrato" and base_freq is not None:
        # simple vibrato via frequency modulation around base_freq
        # re-synthesize: f(t) = base_freq * (1 + depth * sin(2π r t) * 0.02)
        inst = np.sin(2*np.pi*(base_freq*(1 + 0.02*lfo))*t).astype(np.float32)
        # try to preserve original timbre by mixing
        return 0.5*signal + 0.5*inst
    return signal

def to_wav_buffer(audio, sr=SR):
    # prevent NaNs
    audio = np.nan_to_num(audio)
    # normalize
    peak = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
    audio = (audio / peak * 0.98).astype(np.float32)
    audio_i16 = np.int16(audio * 32767)
    buf = io.BytesIO()
    write(buf, sr, audio_i16)
    buf.seek(0)
    return buf

# =========================
# ------ Music Utils ------
# =========================
NOTE_TO_SEMITONE = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
    "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
MAJOR_STEPS = np.array([0,2,4,5,7,9,11])
MINOR_STEPS = np.array([0,2,3,5,7,8,10])

def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69)/12))

def key_scale(root_name, mode="Major"):
    root = NOTE_TO_SEMITONE[root_name]
    steps = MAJOR_STEPS if mode == "Major" else MINOR_STEPS
    return (root + steps) % 12

def triad_degrees(mode="Major"):
    # Return common progressions as degree indices (0-based)
    if mode == "Major":
        return {
            "I–V–vi–IV": [0,4,5,3],
            "ii–V–I–vi": [1,4,0,5],
            "I–IV–V–I": [0,3,4,0],
        }
    else:
        # natural minor-ish choices
        return {
            "i–VI–III–VII": [0,5,2,6],
            "i–iv–VI–V": [0,3,5,4],
            "i–VII–VI–VII": [0,6,5,6],
        }

def diatonic_triad(root_name, mode, degree, octave=4):
    """Return 3 MIDI notes of the diatonic triad on scale degree (0..6)."""
    steps = MAJOR_STEPS if mode == "Major" else MINOR_STEPS
    # triad degrees: d, d+2, d+4 (mod 7)
    degs = [(degree)%7, (degree+2)%7, (degree+4)%7]
    root_semitone = NOTE_TO_SEMITONE[root_name]
    notes = []
    base_midi = 12*(octave+1)  # C4 ~ 60 if root=C
    for i, dg in enumerate(degs):
        semitone_offset = int(root_semitone + steps[dg])
        midi = base_midi + semitone_offset
        # small voicing spread
        midi += 12 * (i//2)  # lift the 5th one octave sometimes
        notes.append(midi)
    return notes

def random_melody_note(root_name, mode, octave=5, allowed=[0,1,2,3,4,5,6]):
    steps = MAJOR_STEPS if mode == "Major" else MINOR_STEPS
    deg = np.random.choice(allowed)
    midi = 12*(octave+1) + NOTE_TO_SEMITONE[root_name] + int(steps[deg])
    return midi

# =========================
# ---------- UI ----------
# =========================
st.set_page_config(page_title="Synthy", page_icon="🎹", layout="centered")
st.title("🎹 Experimental Synth.py")

tab1, tab2 = st.tabs(["🎛️ Sound Design Fundamentals", "🎼 Generative Sequences"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("ADSR Envelope")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        attack = st.slider("Attack (s)", 0.01, 2.0, 0.2, 0.01)
    with colB:
        decay = st.slider("Decay (s)", 0.01, 2.0, 0.3, 0.01)
    with colC:
        sustain = st.slider("Sustain Level", 0.0, 1.0, 0.6, 0.01)
    with colD:
        release = st.slider("Release (s)", 0.01, 2.0, 0.5, 0.01)

    st.subheader("Tone")
    c1, c2 = st.columns(2)
    with c1:
        wave_type = st.selectbox("Waveform", ["Sine", "Square", "Saw", "Triangle"])
        freq = st.slider("Frequency (Hz)", 50, 2000, 440)
        duration = st.slider("Duration (s)", 0.3, 5.0, 2.0, 0.1)
    with c2:
        st.markdown("**LFO**")
        lfo_rate = st.slider("LFO Rate (Hz)", 0.1, 20.0, 5.0, 0.1)
        lfo_depth = st.slider("LFO Depth", 0.0, 1.0, 0.0, 0.01)
        lfo_mode = st.selectbox("LFO Mode", ["Tremolo", "Vibrato"])

    st.subheader("Filter")
    filter_on = st.checkbox("Enable Filter", value=False)
    colF1, colF2 = st.columns(2)
    if filter_on:
        with colF1:
            filter_type = st.radio("Filter Type", ["low", "high"], horizontal=True)
        with colF2:
            cutoff = st.slider("Cutoff (Hz)", 100, 8000, 1200)
    else:
        filter_type = "low"
        cutoff = 1200

    # ---------- Live Visualizers ----------
    # ADSR curve (independent of freq)
    st.markdown("### Visualizers")
    vis_t = np.linspace(0, duration, int(SR*duration), endpoint=False)
    env = adsr_envelope(len(vis_t), attack, decay, sustain, release)

    fig_env, ax_env = plt.subplots(figsize=(6,2.2))
    ax_env.plot(vis_t, env)
    ax_env.set_title("ADSR Envelope")
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Amplitude")
    ax_env.set_ylim([-0.05, 1.05])
    ax_env.grid(True, alpha=0.3)
    st.pyplot(fig_env, clear_figure=True)

    # Waveform preview (first 300 ms)
    preview_ms = min(0.3, duration)
    pt = np.linspace(0, preview_ms, int(SR*preview_ms), endpoint=False)
    pwave = gen_wave(freq, pt, wave_type)
    pwave = apply_lfo(pwave, pt, lfo_rate, lfo_depth, lfo_mode, base_freq=freq, wave_type=wave_type)
    pwave = pwave * adsr_envelope(len(pt), min(attack, preview_ms*0.5), min(decay, preview_ms*0.3), sustain, min(release, preview_ms*0.2))
    if filter_on:
        pwave = butter_filter(pwave, cutoff, SR, filter_type)
    fig_wav, ax_wav = plt.subplots(figsize=(6,2.2))
    ax_wav.plot(pt, pwave)
    ax_wav.set_title("Waveform Preview (first 300 ms)")
    ax_wav.set_xlabel("Time (s)")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.grid(True, alpha=0.3)
    st.pyplot(fig_wav, clear_figure=True)

    @st.cache_data
    def render_single(attack, decay, sustain, release, freq, duration, wave_type,
                      lfo_rate, lfo_depth, lfo_mode, filter_on, cutoff, filter_type):
        t = np.linspace(0, duration, int(SR*duration), endpoint=False)
        sig = gen_wave(freq, t, wave_type)
        sig = apply_lfo(sig, t, lfo_rate, lfo_depth, lfo_mode, base_freq=freq, wave_type=wave_type)
        env = adsr_envelope(len(t), attack, decay, sustain, release)
        sig = sig * env
        if filter_on:
            sig = butter_filter(sig, cutoff, SR, filter_type)
        return to_wav_buffer(sig)

    if st.button("▶️ Play Tone", use_container_width=True):
        wav_buf = render_single(attack, decay, sustain, release, freq, duration, wave_type,
                                lfo_rate, lfo_depth, lfo_mode, filter_on, cutoff, filter_type)
        st.audio(wav_buf, format="audio/wav")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Key & Timing")
    colK1, colK2, colK3 = st.columns(3)
    with colK1:
        root = st.selectbox("Root", ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"], index=0)
    with colK2:
        mode = st.selectbox("Mode", ["Major", "Minor"], index=0)
    with colK3:
        tempo = st.slider("Tempo (BPM)", 60, 160, 100)

    seq_duration = st.slider("Sequence Duration (s)", 10, 20, 14)
    wave_type_seq = st.selectbox("Waveform", ["Sine", "Square", "Saw", "Triangle"], index=2)

    st.markdown("**ADSR for Notes**")
    cA, cD, cS, cR = st.columns(4)
    with cA:
        A2 = st.slider("Attack (s)", 0.01, 1.0, 0.03, 0.01, key="A2")
    with cD:
        D2 = st.slider("Decay (s)", 0.01, 1.0, 0.12, 0.01, key="D2")
    with cS:
        S2 = st.slider("Sustain", 0.0, 1.0, 0.7, 0.01, key="S2")
    with cR:
        R2 = st.slider("Release (s)", 0.01, 1.0, 0.2, 0.01, key="R2")

    st.markdown("**LFO & Filter**")
    cL1, cL2, cL3 = st.columns(3)
    with cL1:
        lfo_rate2 = st.slider("LFO Rate (Hz)", 0.1, 12.0, 4.0, 0.1, key="lfo2r")
    with cL2:
        lfo_depth2 = st.slider("LFO Depth", 0.0, 1.0, 0.15, 0.01, key="lfo2d")
    with cL3:
        lfo_mode2 = st.selectbox("LFO Mode", ["Tremolo", "Vibrato"], index=0, key="lfo2m")

    filter_on2 = st.checkbox("Enable Filter", value=True, key="f2on")
    if filter_on2:
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            filter_type2 = st.radio("Filter Type", ["low", "high"], horizontal=True, key="f2t")
        with fcol2:
            cutoff2 = st.slider("Cutoff (Hz)", 200, 8000, 1600, key="f2c")
    else:
        filter_type2 = "low"
        cutoff2 = 1600

    # Choose chord progression
    progs = triad_degrees(mode)
    prog_name = st.selectbox("Chord Progression", list(progs.keys()))
    degrees = progs[prog_name]  # list of degree indices

    # Rhythm / density
    st.markdown("**Rhythm**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        beats_per_chord = st.slider("Beats per Chord", 2, 8, 4)
    with cc2:
        melody_note_div = st.selectbox("Melody Note Length", ["1 beat","1/2 beat","1/4 beat"], index=1)
    with cc3:
        melody_density = st.slider("Melody Density (0=sparse,1=busy)", 0.0, 1.0, 0.7, 0.05)

    # Generate sequence button
    @st.cache_data
    def render_sequence(root, mode, tempo, seq_duration, wave_type, A2,D2,S2,R2,
                        lfo_rate2, lfo_depth2, lfo_mode2, filter_on2, cutoff2, filter_type2,
                        prog_name, degrees, beats_per_chord, melody_note_div, melody_density):
        total_samples = int(SR*seq_duration)
        t_global = np.arange(total_samples)/SR
        audio = np.zeros(total_samples, dtype=np.float32)

        spb = 60.0/tempo  # seconds per beat
        chord_len_s = beats_per_chord * spb
        chord_positions = []
        # Build chord timeline cycling through chosen progression
        time_cursor = 0.0
        prog_idx = 0
        while time_cursor < seq_duration - 1e-6:
            chord_positions.append((time_cursor, degrees[prog_idx % len(degrees)]))
            time_cursor += chord_len_s
            prog_idx += 1

        # Add chords (lower octave)
        for start_s, deg in chord_positions:
            chord_midis = diatonic_triad(root, mode, deg, octave=3)  # lower register
            for midi in chord_midis:
                f = midi_to_freq(midi)
                dur = min(chord_len_s, seq_duration - start_s)
                n_samples = int(dur*SR)
                if n_samples <= 0: 
                    continue
                start_i = int(start_s*SR)
                end_i = start_i + n_samples
                tt = np.arange(n_samples)/SR
                sig = gen_wave(f, tt, wave_type)
                sig = apply_lfo(sig, tt, lfo_rate2, lfo_depth2, lfo_mode2, base_freq=f, wave_type=wave_type)
                env = adsr_envelope(n_samples, A2, D2, S2, R2)
                sig = sig * env * 0.35  # keep chords a bit lower
                audio[start_i:end_i] += sig.astype(np.float32)

        # Melody generator on top (upper octave, diatonic)
        if melody_note_div == "1 beat":
            note_len_s = spb
        elif melody_note_div == "1/2 beat":
            note_len_s = spb/2.0
        else:
            note_len_s = spb/4.0

        tpos = 0.0
        rng = np.random.default_rng(42)  # deterministic for caching; remove/seed differently for variety
        allowed_degrees = list(range(7))
        while tpos < seq_duration - 1e-6:
            # probabilistic rest
            if rng.random() < (1.0 - melody_density):
                tpos += note_len_s
                continue
            # bias melody degree towards chord tones sometimes
            current_deg = degrees[(int(tpos//chord_len_s)) % len(degrees)]
            options = [current_deg, (current_deg+2)%7, (current_deg+4)%7] + allowed_degrees
            deg_choice = rng.choice(options)
            midi = 12*(5+1) + NOTE_TO_SEMITONE[root] + int((MAJOR_STEPS if mode=="Major" else MINOR_STEPS)[deg_choice])
            f = midi_to_freq(midi)
            dur = min(note_len_s, seq_duration - tpos)
            n_samples = int(dur*SR)
            start_i = int(tpos*SR)
            end_i = start_i + n_samples
            tt = np.arange(n_samples)/SR
            sig = gen_wave(f, tt, wave_type)
            sig = apply_lfo(sig, tt, lfo_rate2, lfo_depth2, lfo_mode2, base_freq=f, wave_type=wave_type)
            env = adsr_envelope(n_samples, max(0.005, A2*0.6), max(0.005, D2*0.7), S2, max(0.04, R2*0.6))
            sig = sig * env * 0.55
            audio[start_i:end_i] += sig.astype(np.float32)
            tpos += note_len_s

        # Optional filtering over the full mix
        if filter_on2:
            audio = butter_filter(audio, cutoff2, SR, filter_type2)

        return to_wav_buffer(audio), audio, chord_positions

    if st.button("🎶 Generate Sequence", use_container_width=True):
        wav_buf2, audio2, chord_pos = render_sequence(
            root, mode, tempo, seq_duration, wave_type_seq, A2,D2,S2,R2,
            lfo_rate2, lfo_depth2, lfo_mode2, filter_on2, cutoff2, filter_type2,
            prog_name, degrees, beats_per_chord, melody_note_div, melody_density
        )
        st.audio(wav_buf2, format="audio/wav")

        # Simple visualization of chord timeline (labels)
        labels = [f"Deg {d+1}" for _, d in chord_pos]
        times = [t for t, _ in chord_pos]
        st.markdown("**Chord Timeline**")
        fig_tl, ax_tl = plt.subplots(figsize=(6, 1.8))
        for i, (start, lab) in enumerate(zip(times, labels)):
            ax_tl.axvspan(start, min(start + beats_per_chord*(60/tempo), seq_duration),
                          alpha=0.15)
            ax_tl.text(start + 0.1, 0.5, lab, va="center", ha="left")
        ax_tl.set_xlim(0, seq_duration)
        ax_tl.set_ylim(0, 1)
        ax_tl.set_yticks([])
        ax_tl.set_xlabel("Time (s)")
        ax_tl.set_title(f"Progression: {prog_name}")
        st.pyplot(fig_tl, clear_figure=True)

        # Waveform snapshot
        st.markdown("**Waveform Snapshot (first 2 seconds)**")
        snap_len = int(min(2.0, seq_duration)*SR)
        fig_ws, ax_ws = plt.subplots(figsize=(6, 2.0))
        ax_ws.plot(np.arange(snap_len)/SR, audio2[:snap_len])
        ax_ws.set_xlabel("Time (s)")
        ax_ws.set_ylabel("Amp")
        ax_ws.grid(True, alpha=0.3)
        st.pyplot(fig_ws, clear_figure=True)



