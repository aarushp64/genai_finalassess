"""
NeuroTutor AI - Concept Image Studio Page
Author: Aarush Parate (Roll No: 1)
Unit IV: Generative AI Tools and Techniques
Converts conceptual topics into rich educational diffusion prompts and generates download-ready infographics.
"""

import streamlit as st
import random
from modules.image_generator import ImageGenerator
from modules.safety_guardrails import SafetyGuardrails


def render_image_studio():
    """Renders the Multimodal Concept Image Studio page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit IV: Multimodal Generation Studio</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">AI Concept Illustration Generator</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Transform complex mathematical and algorithmic concepts into high-resolution educational diagrams 
                using automated prompt expansion and hosted diffusion models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_inp, col_seed = st.columns([3, 1])

    with col_inp:
        predefined_concepts = [
            "Transformer Architecture & Attention Blocks",
            "Generative Adversarial Network (GAN) Architecture",
            "Backpropagation & Gradient Descent Flow",
            "Deep Convolutional Neural Network Layers",
            "Reinforcement Learning Agent-Environment Loop",
            "Diffusion Denoising Probabilistic Model",
        ]
        selected_preset = st.selectbox(
            "Select a Core Concept Preset (or choose Custom):",
            options=predefined_concepts + ["Custom Concept..."],
        )

        if selected_preset == "Custom Concept...":
            concept_text = st.text_input("Enter your custom AI/ML concept:", value="Multi-Head Latent Attention")
        else:
            concept_text = selected_preset

    with col_seed:
        seed = st.number_input("Diffusion Seed:", min_value=1, max_value=999999, value=42, step=1)

    # Show prompt expansion preview
    enhanced_prompt = ImageGenerator.enhance_concept_prompt(concept_text)
    with st.expander("🔍 View Automated Prompt Enhancement (Diffusion Prompt):"):
        st.code(enhanced_prompt, language="text")

    st.write("")
    if st.button("🎨 Generate Educational Illustration", use_container_width=True):
        is_safe, msg = SafetyGuardrails.validate_user_prompt(concept_text)
        if not is_safe:
            st.error(f"🛡️ Safety Alert: {msg}")
            return

        with st.spinner("Generating educational illustration via hosted diffusion endpoint..."):
            result = ImageGenerator.generate_illustration(concept_text, seed=seed)
            st.session_state["latest_studio_image"] = result
            st.session_state["latest_studio_concept"] = concept_text

    if "latest_studio_image" in st.session_state:
        res = st.session_state["latest_studio_image"]
        img_bytes = res["image_bytes"]
        provider = res["provider"]

        st.write("")
        st.markdown("### 🖼️ Generated Educational Diagram")
        st.caption(f"🤖 **Diffusion Engine:** {provider} | 🔢 **Seed:** {seed}")

        st.image(img_bytes, caption=f"Concept Illustration: {st.session_state.get('latest_studio_concept')}", use_container_width=True)

        # Download button
        st.download_button(
            label="⬇️ Download High-Resolution Diagram (.png)",
            data=img_bytes,
            file_name=f"neurotutor_{concept_text.lower().replace(' ', '_')[:25]}.png",
            mime="image/png",
            use_container_width=True,
        )
