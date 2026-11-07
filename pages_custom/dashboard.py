"""
NeuroTutor AI - Dashboard Page
Author: Aarush Parate (Roll No: 1)
Project: P1 - AI-ML-DL Concept Tutor
"""

import streamlit as st
import plotly.graph_objects as go
import config


def render_dashboard():
    """Renders the central executive dashboard for NeuroTutor AI."""
    st.markdown(
        f"""
        <div class="neuro-card-accent">
            <span class="badge-tag">Academic CA Mini-Project P1</span>
            <h1 style="margin: 0.2rem 0; font-size: 2.2rem; color: #F2F2F2;">NeuroTutor AI</h1>
            <p style="color: #AEB7BF; font-size: 1.05rem; margin-top: 0.4rem; max-width: 850px;">
                An intelligent multi-level educational tutor and interactive research lab spanning 
                Foundations of Generative AI, GANs, Large Language Models, Multimodal Systems, and Responsible AI.
            </p>
            <div style="display: flex; gap: 1.5rem; margin-top: 0.8rem; font-size: 0.85rem; color: #8EAABF;">
                <span>👤 <strong>Student:</strong> {config.STUDENT_NAME}</span>
                <span>🔢 <strong>Roll No:</strong> {config.STUDENT_ROLL}</span>
                <span>📌 <strong>Course:</strong> {config.COURSE_NAME}</span>
                <span>🎯 <strong>Problem Code:</strong> {config.PROJECT_CODE}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Performance / Syllabus Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-val">5 / 5</div>
                <div class="stat-label">Syllabus Units Implemented</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-val">3 Levels</div>
                <div class="stat-label">School • UG • Expert</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-val">72 Heads</div>
                <div class="stat-label">Real Attention Weights</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-val">93.3%</div>
                <div class="stat-label">Grounding Verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Three Learning Levels Showcase
    st.markdown("### 🎓 Adaptive Three-Tier Learning Levels")
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown(
            """
            <div class="neuro-card">
                <span class="badge-tag badge-school">Tier 1: Intuition</span>
                <h4 style="margin: 0.3rem 0; color: #81C784;">School Level</h4>
                <p style="color: #AEB7BF; font-size: 0.9rem;">
                    Everyday real-world analogies (baking recipes, sports metaphors, LEGO sorting), intuitive conceptual metaphors, and zero intimidating mathematical notation.
                </p>
                <div style="font-size: 0.8rem; color: #728EA6;">Focus: Conceptual clarity & wonder</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l2:
        st.markdown(
            """
            <div class="neuro-card">
                <span class="badge-tag badge-ug">Tier 2: Applied</span>
                <h4 style="margin: 0.3rem 0; color: #64B5F6;">Undergraduate Level</h4>
                <p style="color: #AEB7BF; font-size: 0.9rem;">
                    Algorithmic breakdowns, mathematical intuition, pseudocode, network architectures, complexity analysis, and practical PyTorch implementation details.
                </p>
                <div style="font-size: 0.8rem; color: #728EA6;">Focus: Computational rigor & engineering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l3:
        st.markdown(
            """
            <div class="neuro-card">
                <span class="badge-tag badge-expert">Tier 3: Research</span>
                <h4 style="margin: 0.3rem 0; color: #BA68C8;">Expert Level</h4>
                <p style="color: #AEB7BF; font-size: 0.9rem;">
                    Formal mathematical derivations, loss manifolds, attention complexity bottlenecks (FlashAttention), theoretical scaling laws, and frontier research insights.
                </p>
                <div style="font-size: 0.8rem; color: #728EA6;">Focus: Mathematical frontiers & scaling</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Five Units Feature Map
    st.markdown("### 🗺️ Mandatory Five-Unit Syllabus Architecture")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown(
            """
            <div class="neuro-card">
                <h4 style="color: #F2F2F2; margin-top: 0;">Unit I — Foundations of Generative AI</h4>
                <ul style="color: #AEB7BF; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Prompt Engineering Lab:</strong> Side-by-side Zero-Shot, Few-Shot, and Chain-of-Thought comparison.</li>
                    <li><strong>Temperature Softmax Control:</strong> 0.0 to 1.5 temperature slider with dynamic logit-to-probability math.</li>
                    <li><strong>Generative vs Discriminative:</strong> P(X, Y) vs P(Y|X) visual comparisons and decision boundaries.</li>
                    <li><strong>GenAI Lifecycle:</strong> Interactive end-to-end data preparation, training, and deployment pipeline.</li>
                    <li><strong>Latent Space Projection:</strong> 2D concept embedding clustering and exploration.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="neuro-card">
                <h4 style="color: #F2F2F2; margin-top: 0;">Unit III — GPT, BERT & Multimodal Models</h4>
                <ul style="color: #AEB7BF; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Academic Notes Ingestion:</strong> Multi-format parser (PDF, DOCX, TXT) with chunking.</li>
                    <li><strong>Sentence-BERT Retrieval:</strong> all-MiniLM-L6-v2 dense 384-d vectors with FAISS cosine index.</li>
                    <li><strong>Grounded Context Answers:</strong> LLM answers grounded in notes with citation references.</li>
                    <li><strong>BLIP Diagram Explainer:</strong> Multimodal visual architecture parsing and object detection.</li>
                    <li><strong>GPT vs BERT Comparison:</strong> Causal autoregressive decoder vs bidirectional encoder MLM.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_u2:
        st.markdown(
            """
            <div class="neuro-card">
                <h4 style="color: #F2F2F2; margin-top: 0;">Unit II — GANs & Large Language Models</h4>
                <ul style="color: #AEB7BF; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Conversational Concept Tutor:</strong> Multi-turn dialogue with School, UG, and Expert levels.</li>
                    <li><strong>Bounded Memory Manager:</strong> Sliding-window session memory preserving context.</li>
                    <li><strong>Real Attention Heatmap:</strong> Extracts DistilBERT attention weights across 6 layers & 12 heads.</li>
                    <li><strong>PyTorch GAN Explorer:</strong> Interactive live adversarial minimax training on continuous distributions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="neuro-card">
                <h4 style="color: #F2F2F2; margin-top: 0;">Units IV & V — Tools, Techniques & Responsible AI</h4>
                <ul style="color: #AEB7BF; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Multimodal Studio:</strong> Hosted diffusion AI illustrations and gTTS audio summaries.</li>
                    <li><strong>PyTorch MLP Demo:</strong> Transparent forward activations, loss, and backprop gradients.</li>
                    <li><strong>Grounding & Hallucination Checker:</strong> Verifies claims with Supported / Partial / Unsupported statuses.</li>
                    <li><strong>Safety Guardrails:</strong> Regex & heuristic prompt injection and harmful content defense.</li>
                    <li><strong>Measurable Evaluation & Model Card:</strong> Labelled benchmark evaluation and comprehensive MODEL_CARD.md.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Technology Stack Table
    st.markdown("### ⚙️ Production Technology Stack")
    st.markdown(
        """
        | Component | Technology / Library | Execution Mode | Role in NeuroTutor AI |
        | :--- | :--- | :--- | :--- |
        | **Frontend UI** | Streamlit 1.51 + Custom CSS + Plotly | Local Web App | Interactive modern SaaS interface |
        | **Core Framework** | Python 3.13 + PyTorch 2.9 (CPU) | Local Execution | GAN training, MLP autograd, attention computation |
        | **Transformer Models** | Hugging Face `transformers` | Local Execution | DistilBERT 6-layer / 12-head attention extraction |
        | **Dense Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Local Execution | 384-dimensional dense semantic vectors |
        | **Vector Search** | FAISS CPU (`IndexFlatIP`) | In-Memory Local | Fast cosine similarity search over notes |
        | **LLM Inference** | Google Gemini 2.5 Flash / Groq / Fallback | API + Local Fallback | Conceptual explanation, synthesis & reasoning |
        | **Vision Multimodal** | Gemini Vision / BLIP Heuristics | API + Local Fallback | Educational schematic diagram interpretation |
        | **Image Generation** | Hosted Diffusion API (Pollinations) | Hosted HTTP Inference | Automated prompt expansion & concept illustration |
        | **Audio Speech** | `gTTS` (Google Text-to-Speech) | Python Library | Markdown-cleaned voice summary audio generation |
        | **Document Processing**| `pypdf`, `python-docx` | Local Execution | Clean text extraction & semantic chunking |
        """
    )
