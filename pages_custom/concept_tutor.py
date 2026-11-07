"""
NeuroTutor AI - Concept Tutor Page
Author: Aarush Parate (Roll No: 1)
Unit II: Conversational AI Tutor & Session Memory
"""

import streamlit as st
from modules.llm_engine import LLMEngine
from modules.conversation_memory import ConversationMemoryManager
from modules.safety_guardrails import SafetyGuardrails
from modules.audio_generator import AudioSummaryGenerator
import config


def render_concept_tutor(llm: LLMEngine, memory: ConversationMemoryManager):
    """Renders the conversational AI Concept Tutor interface."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit II: Conversational AI</span>
            <h2 style="margin: 0.2rem 0; color: #F2F2F2;">Conversational Concept Tutor</h2>
            <p style="color: #AEB7BF; font-size: 0.95rem; margin-top: 0.3rem;">
                Ask any question about AI, Machine Learning, or Deep Learning. 
                Select your knowledge level and receive tailored analogies, algorithms, or research insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Controls Bar: Level & Temperature
    col_level, col_temp = st.columns([2, 1])

    with col_level:
        level = st.selectbox(
            "Select Knowledge Level:",
            options=list(config.LEARNING_LEVELS.keys()),
            index=1,
            help="Adapts the cognitive vocabulary, mathematical formality, and analogies.",
        )
        level_info = config.LEARNING_LEVELS[level]
        st.caption(f"💡 **Level Guidance:** {level_info['description']}")

    with col_temp:
        temp = st.slider(
            "Temperature (Randomness):",
            min_value=config.TEMPERATURE_MIN,
            max_value=config.TEMPERATURE_MAX,
            value=config.DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls probability flattening in softmax: 0.0 is deterministic/focused, >1.0 is exploratory.",
        )

    st.write("")

    # Display Conversation History
    history = memory.get_or_create_history(st.session_state)

    if history:
        st.markdown("### 💬 Conversation History")
        for idx, turn in enumerate(history):
            badge_class = "badge-school" if "School" in turn["level"] else ("badge-ug" if "Undergraduate" in turn["level"] else "badge-expert")
            
            # User message
            st.markdown(
                f"""
                <div class="chat-user">
                    <span style="font-size: 0.75rem; color: #8EAABF; font-weight: 600;">You</span>
                    <div style="color: #F2F2F2; margin-top: 0.2rem;">{turn['user']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tutor message
            st.markdown(
                f"""
                <div class="chat-assistant">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #8EAABF;">NeuroTutor AI</span>
                        <span class="badge-tag {badge_class}" style="margin: 0;">{turn['level']}</span>
                    </div>
                    <div style="color: #F2F2F2; line-height: 1.6;">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(turn["assistant"])
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Clear History Button
        col_clr, _ = st.columns([1, 4])
        with col_clr:
            if st.button("🗑️ Reset Conversation", use_container_width=True):
                memory.clear_history(st.session_state)
                st.rerun()

    st.write("")

    # Question Input Form
    with st.form("tutor_input_form", clear_on_submit=True):
        st.markdown("#### ❓ Ask Your Question / Follow-up")
        user_question = st.text_input(
            "Enter an AI/ML/DL question (e.g., 'What is Backpropagation?', 'How do Transformers work?', 'Explain GAN loss'):",
            placeholder="Type your question here...",
        )
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            submitted = st.form_submit_button("🚀 Generate Answer", use_container_width=True)

    if submitted:
        if not user_question.strip():
            st.warning("Please enter a question before submitting.")
            return

        # 1. Safety Guardrails Validation
        is_safe, msg = SafetyGuardrails.validate_user_prompt(user_question)
        if not is_safe:
            st.error(f"🛡️ Safety Guardrail Alert: {msg}")
            return

        with st.spinner(f"Generating {level} explanation at T={temp}..."):
            # 2. Compile bounded memory context
            prompt_with_context = memory.compile_context_prompt(
                st.session_state, user_question, level
            )

            # 3. Call LLM Engine
            response = llm.generate(
                prompt=prompt_with_context,
                temperature=temp,
                level=level,
            )

            answer_text = response["text"]
            latency = response.get("latency", 0.0)
            tokens = response.get("tokens_approx", 0)
            provider = response.get("provider", "Engine")

            # 4. Save to session conversation memory
            memory.add_turn(
                session_state=st.session_state,
                user_message=user_question,
                assistant_message=answer_text,
                level=level,
                metadata={"latency": latency, "tokens": tokens, "temperature": temp},
            )

            # Store latest response in session state for audio synthesis
            st.session_state["latest_tutor_response"] = answer_text
            st.session_state["latest_tutor_question"] = user_question
            st.session_state["latest_provider"] = provider
            st.session_state["latest_latency"] = latency

            st.rerun()

    # Audio synthesis option for latest explanation
    if "latest_tutor_response" in st.session_state and st.session_state["latest_tutor_response"]:
        st.write("")
        st.markdown("---")
        st.markdown("### 🔊 Listen to Spoken Audio Summary")
        c_aud1, c_aud2 = st.columns([2, 1])
        with c_aud1:
            accent = st.selectbox(
                "Voice Accent:",
                options=[("US English", "com"), ("UK English", "co.uk"), ("Indian English", "co.in")],
                format_func=lambda x: x[0],
                key="tutor_accent_select",
            )
        with c_aud2:
            st.write("")
            gen_audio_btn = st.button("🎧 Synthesize Audio", use_container_width=True)

        if gen_audio_btn:
            with st.spinner("Synthesizing audio explanation via gTTS..."):
                audio_bytes, audio_msg = AudioSummaryGenerator.generate_speech(
                    st.session_state["latest_tutor_response"], accent_tld=accent[1]
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="⬇️ Download Audio (.mp3)",
                        data=audio_bytes,
                        file_name="neurotutor_concept_summary.mp3",
                        mime="audio/mp3",
                    )
                else:
                    st.error(audio_msg)
