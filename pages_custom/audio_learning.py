"""
NeuroTutor AI - Audio Learning Page
Author: Aarush Parate (Roll No: 1)
Unit IV: Generative AI Tools and Techniques
Text-to-Speech audio narration engine with accent controls, playback, and MP3 download.
"""

import streamlit as st
from modules.audio_generator import AudioSummaryGenerator


def render_audio_learning():
    """Renders the Audio Learning Studio page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit IV: Generative AI Audio Tools</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">Audio Learning & Speech Synthesis</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Listen to spoken audio recaps of complex AI/ML concepts. 
                Cleans mathematical notation and markdown for smooth acoustic clarity.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preload latest explanation if available
    default_text = st.session_state.get(
        "latest_tutor_response",
        "The self-attention mechanism in Transformers computes pairwise relationships between all words in a sentence simultaneously. "
        "Unlike recurrent neural networks which process tokens one by one, attention allows the model to capture long-range dependencies in constant sequential operations.",
    )

    c_tx, c_ctrl = st.columns([3, 1.2])

    with c_tx:
        text_to_speak = st.text_area(
            "Text to Convert to Speech:",
            value=default_text,
            height=220,
            help="Markdown tags and LaTeX math will be automatically sanitized for natural speech.",
        )

    with c_ctrl:
        st.markdown("#### 🎙️ Voice Settings")
        accent = st.selectbox(
            "Regional Accent:",
            options=[
                ("US English", "com"),
                ("UK English (British)", "co.uk"),
                ("Indian English", "co.in"),
                ("Australian English", "com.au"),
                ("Canadian English", "ca"),
            ],
            format_func=lambda x: x[0],
        )

        speed_slow = st.checkbox("Slow Narration Speed", value=False)

        st.write("")
        gen_audio = st.button("🔊 Synthesize Speech", use_container_width=True)

    if gen_audio:
        if not text_to_speak.strip():
            st.warning("Please provide text to synthesize.")
            return

        with st.spinner("Synthesizing audio via gTTS..."):
            audio_bytes, msg = AudioSummaryGenerator.generate_speech(
                text=text_to_speak,
                accent_tld=accent[1],
                slow=speed_slow,
            )

            if audio_bytes:
                st.session_state["audio_player_bytes"] = audio_bytes
                st.success("Speech synthesized successfully!")
            else:
                st.error(msg)

    if "audio_player_bytes" in st.session_state:
        audio_data = st.session_state["audio_player_bytes"]
        st.write("")
        st.markdown("### 🎧 Audio Playback")
        st.audio(audio_data, format="audio/mp3")

        c_dl, _ = st.columns([1, 3])
        with c_dl:
            st.download_button(
                label="⬇️ Download Audio File (.mp3)",
                data=audio_data,
                file_name="neurotutor_spoken_summary.mp3",
                mime="audio/mp3",
                use_container_width=True,
            )
