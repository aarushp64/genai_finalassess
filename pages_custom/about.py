"""
NeuroTutor AI - About & Model Card Page
Author: Aarush Parate (Roll No: 1)
Section 4 & 5: Comprehensive Model Card and Transparency Manifest
"""

import streamlit as st
import config


def render_about():
    """Renders the comprehensive About, Model Card, and Transparency page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Academic CA Mini-Project P1</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">About NeuroTutor AI & Model Card</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Official academic submission, comprehensive model card, and model transparency manifest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_card, tab_models, tab_meta = st.tabs(["📋 Formal Model Card", "🔍 Model Transparency Table", "🎓 Student Submission Details"])

    with tab_card:
        st.markdown(
            f"""
            ### 📄 Formal Model Card: NeuroTutor AI
            *Structured in accordance with Mitchell et al. (2019) Model Cards for Model Reporting.*

            #### 1. Model Details
            - **Application Name:** NeuroTutor AI
            - **Author:** {config.STUDENT_NAME} (Roll Number: {config.STUDENT_ROLL})
            - **Problem Code:** {config.PROJECT_CODE} — AI–ML–DL Concept Tutor
            - **Course:** {config.COURSE_NAME} (Individual Continuous Assessment Mini-Project)
            - **Frameworks:** Python 3.13, PyTorch 2.9, Hugging Face Transformers 4.57, Sentence-Transformers 5.1, Streamlit 1.51, FAISS CPU 1.12.

            #### 2. Intended Use
            - **Primary Domain:** Educational pedagogy in Artificial Intelligence, Machine Learning, and Deep Learning for undergraduate computer science education.
            - **Target Audience:** School students (Tier 1 intuitive analogies), Undergraduate students (Tier 2 algorithmic and mathematical intuition), and AI researchers/experts (Tier 3 formal research derivations).
            - **Interactive Capabilities:** Prompt engineering comparison, real attention weight inspection, notes RAG semantic search, image diagram interpretation, diffusion illustration synthesis, and spoken audio summaries.

            #### 3. Out-of-Scope & Prohibited Uses
            - High-stakes autonomous decision-making (medical triage, financial trading, legal judgment).
            - Generation of malicious exploit scripts, weapons manufacturing instructions, or security penetration bypasses.
            - Replacement of formal academic peer review or verbatim exam cheating.

            #### 4. Training & Data Sources
            - **Underlying Models:** Google Gemini 2.5 Flash (pre-trained web corpora and multi-modal datasets), DistilBERT (Toronto Book Corpus + English Wikipedia), Sentence-BERT (1B+ semantic sentence pairs).
            - **Grounding Corpus:** User-uploaded lecture notes (PDF, DOCX, TXT) and pre-packaged syllabus course notes.

            #### 5. Quantitative Performance Summary
            - **Semantic Retrieval Hit Rate:** 70.0% top-3 retrieval across diverse syllabus queries.
            - **Grounding Verification Accuracy:** 93.33% (14/15 correct classifications on labelled benchmark).
            - **Attention Extraction Resolution:** 6 transformer layers &times; 12 attention heads = 72 real attention weight subspaces.
            - **Retrieval Search Latency:** ~12.9 ms average vector search over FAISS CPU index.

            #### 6. Known Limitations
            - **Heuristic Grounding Boundaries:** Rule and lexical overlap methods can underestimate novel semantic paraphrases that lack verbatim word overlap.
            - **Vision Model Ambiguity:** Standard base vision models (e.g. BLIP-1) identify coarse visual objects but may miss microscopic subscript indices on architectural diagram arrows without OCR enhancement.
            - **Hardware Constraints:** Runs efficiently on standard CPU architectures; large diffusion models rely on hosted inference endpoints.
            """
        )

    with tab_models:
        st.markdown("### 🔍 Model & Service Transparency Manifest")
        st.markdown(
            "In strict compliance with **Unit IV Feature 4**, the table below clearly specifies every model, execution mode, and provider used in NeuroTutor AI:"
        )

        st.markdown(
            """
            | Capability | Model / Library Name | Execution Mode | Provider / Host | License |
            | :--- | :--- | :--- | :--- | :--- |
            | **Text Reasoning & LLM** | `gemini-2.5-flash` / `llama-3.3-70b` | Hosted Inference API | Google AI Studio / Groq | Free-Tier API |
            | **Offline Fallback Engine** | NeuroTutor Knowledge Engine | Local Deterministic | Local CPU | MIT License |
            | **Attention Heatmap Extraction** | `distilbert-base-uncased` | **Local PyTorch Execution** | Local Hugging Face Cache | Apache 2.0 |
            | **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | **Local Execution** | Local Cache (384-d) | Apache 2.0 |
            | **Vector Search Engine** | `faiss-cpu` (IndexFlatIP) | **Local In-Memory** | Meta Research | MIT License |
            | **Diagram Vision Analysis** | Gemini 2.5 Flash Vision / BLIP Heuristics | Multimodal API + Fallback | Google AI / Local Heuristic | API / Open |
            | **Concept Illustration** | Hosted Diffusion Inference Pipeline | Hosted Diffusion Endpoint | Pollinations AI Inference | Open Inference |
            | **Procedural Fallback Graphics** | Procedural Vector Infographic Generator | **Local PIL Execution** | Local CPU | Open-Source |
            | **Spoken Audio Synthesis** | `gTTS` (Google Text-to-Speech) | Python Library | Local / Google TTS Endpoint | Open-Source |
            | **Adversarial Minimax GAN** | Custom PyTorch Generative Model | **Local PyTorch Execution** | Local CPU Training | Open-Source |
            | **MLP Autograd Demo** | Custom 3-Layer MLP PyTorch Module | **Local PyTorch Execution** | Local CPU Forward/Backward | Open-Source |
            """
        )

    with tab_meta:
        st.markdown(
            f"""
            ### 🎓 Student Submission Metadata
            - **Student Name:** {config.STUDENT_NAME}
            - **Roll Number:** {config.STUDENT_ROLL}
            - **Allocated Problem Code:** {config.PROJECT_CODE}
            - **Problem Title:** AI–ML–DL Concept Tutor
            - **Course Name:** {config.COURSE_NAME} (Continuous Assessment Mini-Project)
            - **Submission Directory:** `1_Aarush_Parate/`
            - **Primary Documentation Report:** `1_Aarush_Parate_P1.pdf`

            #### 5-Unit Implementation Checklist:
            - [x] **Unit I: Foundations of GenAI** — Prompt Engineering Lab (Zero, Few, CoT), Temperature Softmax Dynamics, Generative vs Discriminative, GenAI Lifecycle, Latent Space.
            - [x] **Unit II: GANs & Large Language Models** — Conversational Concept Tutor (School, UG, Expert), Bounded Memory, Real DistilBERT Attention Heatmap, PyTorch GAN Explorer.
            - [x] **Unit III: GPT, BERT & Multimodal Models** — Academic Notes Ingestion (PDF, DOCX, TXT), Sentence-BERT `all-MiniLM-L6-v2` + FAISS Retrieval, Grounded Answers, BLIP Diagram Explainer, GPT vs BERT Comparison.
            - [x] **Unit IV: Generative AI Tools & Techniques** — Hosted Diffusion Concept Illustration Studio, gTTS Audio Narrator, Educational PyTorch Autograd Demo, Model Transparency Table.
            - [x] **Unit V: Responsible AI and Ethics** — Hallucination Grounding Checker, 15-case Labelled Benchmark Evaluation, Prompt Injection & Safety Guardrails, Ethical Licensing, Formal Model Card.
            """
        )
