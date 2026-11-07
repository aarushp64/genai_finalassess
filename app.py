"""
NeuroTutor AI - Main Application Orchestrator
Author: Aarush Parate (Roll No: 1)
Allocated Problem: P1 - AI-ML-DL Concept Tutor
Course: Generative AI Continuous Assessment (B.Tech CS)
"""

import streamlit as st
import os
from pathlib import Path

import config
from modules.llm_engine import LLMEngine
from modules.conversation_memory import ConversationMemoryManager
from modules.prompt_engineering import PromptEngineeringLab
from modules.attention_visualizer import AttentionVisualizer
from modules.rag_engine import RAGEngine
from modules.image_explainer import ImageExplainer

# Import UI Page Renderers
from pages_custom.dashboard import render_dashboard
from pages_custom.concept_tutor import render_concept_tutor
from pages_custom.prompt_lab import render_prompt_lab
from pages_custom.attention_explorer import render_attention_explorer
from pages_custom.knowledge_base import render_knowledge_base
from pages_custom.diagram_explainer import render_diagram_explainer
from pages_custom.image_studio import render_image_studio
from pages_custom.audio_learning import render_audio_learning
from pages_custom.gan_explorer_page import render_gan_explorer
from pages_custom.responsible_ai import render_responsible_ai
from pages_custom.about import render_about

# 1. Page Configuration
st.set_page_config(
    page_title=f"{config.APP_TITLE} — {config.APP_SUBTITLE}",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Stylesheet
css_path = config.ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# 3. Initialize Shared Engine Singletons in Session State
@st.cache_resource
def get_shared_engines():
    """Initializes heavy backend modules once across user sessions."""
    llm = LLMEngine()
    memory = ConversationMemoryManager(max_turns=config.MAX_HISTORY_TURNS)
    lab = PromptEngineeringLab(llm)
    attn_viz = AttentionVisualizer()
    rag = RAGEngine()
    explainer = ImageExplainer()
    return llm, memory, lab, attn_viz, rag, explainer


llm_engine, memory_mgr, prompt_lab_engine, attention_visualizer, rag_engine, diagram_explainer = (
    get_shared_engines()
)

# 4. Sidebar Navigation & Global Controls
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <h2 style="margin: 0; color: #F2F2F2; font-size: 1.5rem;">🧠 {config.APP_TITLE}</h2>
            <div style="font-size: 0.75rem; color: #8EAABF; margin-top: 0.2rem;">
                AI–ML–DL Multi-Level Concept Tutor
            </div>
            <div style="display: inline-block; background-color: rgba(114, 142, 166, 0.15); border: 1px solid rgba(114, 142, 166, 0.3); border-radius: 9999px; padding: 0.2rem 0.6rem; font-size: 0.7rem; color: #8EAABF; margin-top: 0.5rem;">
                Roll 1 • Aarush Parate • P1
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Navigation")
    pages = [
        "1. Dashboard",
        "2. AI Concept Tutor",
        "3. Prompt Engineering Lab",
        "4. Attention Explorer",
        "5. Knowledge Base & RAG",
        "6. Diagram Explainer",
        "7. Concept Image Studio",
        "8. Audio Learning",
        "9. GAN Explorer",
        "10. Responsible AI",
        "11. About & Model Card",
    ]

    selected_page = st.radio(
        "Go to page:",
        options=pages,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API Configuration Settings (Expandable)
    with st.expander("🔑 API Key Configuration"):
        user_key = st.text_input(
            "Gemini API Key:",
            type="password",
            value=config.GEMINI_API_KEY,
            placeholder="AIzaSy...",
            help="Get your free API key at aistudio.google.com. If omitted, NeuroTutor runs on its high-quality offline educational knowledge engine.",
        )
        if user_key != llm_engine.api_key:
            llm_engine.api_key = user_key
            diagram_explainer.api_key = user_key

        groq_key = st.text_input(
            "Groq API Key (Optional):",
            type="password",
            value=config.GROQ_API_KEY,
            placeholder="gsk_...",
        )
        if groq_key != llm_engine.groq_api_key:
            llm_engine.groq_api_key = groq_key

    # Active Provider Status
    if llm_engine.api_key:
        status_color = "#4CAF50"
        status_text = "Google Gemini 2.5 Flash (Live API)"
    elif llm_engine.groq_api_key:
        status_color = "#2196F3"
        status_text = "Groq Llama 3.3 70B (Live API)"
    else:
        status_color = "#FF9800"
        status_text = "Local Knowledge Engine (Offline Active)"

    st.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #AEB7BF; padding: 0.4rem 0;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; margin-right: 6px;"></span>
            <strong>Engine:</strong> {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.7rem; color: #6C7A89; text-align: center;">
            {config.COURSE_NAME} Continuous Assessment<br>
            B.Tech Computer Science &bull; Roll 1
        </div>
        """,
        unsafe_allow_html=True,
    )

# 5. Route to Selected UI Page
if "1. Dashboard" in selected_page:
    render_dashboard()
elif "2. AI Concept Tutor" in selected_page:
    render_concept_tutor(llm=llm_engine, memory=memory_mgr)
elif "3. Prompt Engineering Lab" in selected_page:
    render_prompt_lab(lab=prompt_lab_engine)
elif "4. Attention Explorer" in selected_page:
    render_attention_explorer(visualizer=attention_visualizer)
elif "5. Knowledge Base & RAG" in selected_page:
    render_knowledge_base(rag=rag_engine, llm=llm_engine)
elif "6. Diagram Explainer" in selected_page:
    render_diagram_explainer(explainer=diagram_explainer)
elif "7. Concept Image Studio" in selected_page:
    render_image_studio()
elif "8. Audio Learning" in selected_page:
    render_audio_learning()
elif "9. GAN Explorer" in selected_page:
    render_gan_explorer()
elif "10. Responsible AI" in selected_page:
    render_responsible_ai()
elif "11. About & Model Card" in selected_page:
    render_about()
