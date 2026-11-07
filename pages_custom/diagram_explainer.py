"""
NeuroTutor AI - Diagram Explainer Page
Author: Aarush Parate (Roll No: 1)
Unit III: GPT, BERT and Multimodal Models
Multimodal interpretation of AI architecture schematics with transparent vision limitation disclosures.
"""

import streamlit as st
import io
from PIL import Image
from modules.image_explainer import ImageExplainer
from modules.image_generator import ImageGenerator
import config


def render_diagram_explainer(explainer: ImageExplainer):
    """Renders the Multimodal Diagram Explainer page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit III: Multimodal Models</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">AI Architecture Diagram Explainer</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Upload architecture schematics, flowcharts, or neural network diagrams to receive automated 
                visual captions, structural component identification, and technical interpretations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_opt1, c_opt2 = st.columns([2, 1])

    with c_opt1:
        uploaded_image = st.file_uploader(
            "Upload an AI/ML Diagram (PNG, JPG, JPEG, WEBP):",
            type=["png", "jpg", "jpeg", "webp"],
        )

    with c_opt2:
        st.markdown("**Or select a built-in architecture diagram:**")
        sample_diagram = st.selectbox(
            "Select Pre-Packaged Diagram:",
            options=[
                "None (Use Uploaded)",
                "Transformer Multi-Head Attention",
                "Generative Adversarial Network (GAN)",
                "Convolutional Neural Network (CNN)",
                "Deep MLP Backpropagation Pipeline",
            ],
            index=1,
        )

    image_bytes = None
    diagram_hint = None

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        diagram_hint = uploaded_image.name
        st.image(image_bytes, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
    elif sample_diagram != "None (Use Uploaded)":
        diagram_hint = sample_diagram
        # Generate procedural reference image for sample
        _, image_bytes = ImageGenerator._create_procedural_fallback_image(sample_diagram)
        st.image(image_bytes, caption=f"Reference Schematic: {sample_diagram}", use_container_width=True)

    st.write("")

    # Knowledge level selector for explanation
    level = st.selectbox(
        "Explanation Level:",
        options=list(config.LEARNING_LEVELS.keys()),
        index=1,
        key="diagram_level_sel",
    )

    if st.button("🔬 Analyze Diagram", use_container_width=True):
        if not image_bytes:
            st.warning("Please upload an image or select a sample diagram first.")
            return

        with st.spinner("Executing multimodal vision inference and architectural parsing..."):
            analysis = explainer.explain_diagram(
                image_bytes=image_bytes,
                user_hint=diagram_hint,
                level=level,
            )
            st.session_state["latest_diagram_analysis"] = analysis

    if "latest_diagram_analysis" in st.session_state:
        analysis = st.session_state["latest_diagram_analysis"]

        st.write("")
        st.markdown("### 📊 Architectural Analysis Results")

        # Transparency Banner
        model_name = analysis.get("model_used", "Vision Model")
        st.caption(f"🤖 **Vision Engine:** {model_name} | 📐 **Resolution:** {analysis.get('image_dimensions', 'N/A')}")

        st.markdown(
            f"""
            <div class="neuro-card">
                {analysis.get('detailed_analysis', 'No analysis output.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Limitation Disclosure Banner (Mandatory requirement in Unit III)
        limitation_text = analysis.get("limitation_disclosure", "")
        if limitation_text:
            st.markdown(
                f"""
                <div class="neuro-card" style="border-left: 4px solid #FF9800; background-color: #354252;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #FFE082;">⚖️ Transparent Limitation Disclosure</div>
                    <div style="font-size: 0.8rem; color: #AAB5C4; margin-top: 0.2rem;">
                        {limitation_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
