"""
NeuroTutor AI - Prompt Engineering Lab Page
Author: Aarush Parate (Roll No: 1)
Unit I: Foundations of Generative AI
Prompt comparison (Zero-Shot, Few-Shot, CoT), Temperature Softmax Dynamics, Generative vs Discriminative, GenAI Lifecycle, and Latent Space.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.prompt_engineering import PromptEngineeringLab
from modules.llm_engine import LLMEngine
from modules.safety_guardrails import SafetyGuardrails
import config


def render_prompt_lab(lab: PromptEngineeringLab):
    """Renders the dedicated Prompt Engineering Lab."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit I: Foundations of GenAI</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">Prompt Engineering Lab</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Compare prompting paradigms (Zero-Shot, Few-Shot, Chain-of-Thought), inspect softmax temperature dynamics, 
                and explore probabilistic latent representations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "⚡ Prompt Comparison",
            "🌡️ Temperature & Softmax Dynamics",
            "⚖️ Generative vs Discriminative",
            "🔄 GenAI Lifecycle",
            "🌌 Latent-Space Explorer",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Prompt Comparison (Zero-Shot, Few-Shot, Chain-of-Thought)
    # -------------------------------------------------------------
    with tab1:
        st.markdown("### 🔬 Compare Prompting Strategies Side-by-Side")
        st.markdown(
            "Observe how different prompt construction patterns alter the LLM's reasoning depth, output structure, token consumption, and response latency for the exact same query."
        )

        col_q, col_lvl, col_tmp = st.columns([3, 1.5, 1])
        with col_q:
            test_concept = st.text_input(
                "Enter Concept to Compare:",
                value="Self-Attention in Transformers",
                help="Concept to test across all 3 prompting strategies.",
            )
        with col_lvl:
            test_level = st.selectbox(
                "Knowledge Level:",
                options=list(config.LEARNING_LEVELS.keys()),
                index=1,
                key="prompt_lab_level",
            )
        with col_tmp:
            test_temp = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.5,
                value=0.7,
                step=0.1,
                key="prompt_lab_temp",
            )

        if st.button("🚀 Run Triple-Technique Comparison", use_container_width=True):
            is_safe, msg = SafetyGuardrails.validate_user_prompt(test_concept)
            if not is_safe:
                st.error(f"🛡️ Security Alert: {msg}")
                return

            with st.spinner("Executing Zero-Shot, Few-Shot, and Chain-of-Thought prompts..."):
                results = lab.compare_prompt_techniques(
                    user_question=test_concept,
                    level=test_level,
                    temperature=test_temp,
                )
                st.session_state["prompt_comparison_results"] = results

        if "prompt_comparison_results" in st.session_state:
            results = st.session_state["prompt_comparison_results"]

            col_zs, col_fs, col_cot = st.columns(3)

            with col_zs:
                st.markdown(
                    """
                    <div class="prompt-box">
                        <span class="badge-tag" style="background-color: rgba(33, 150, 243, 0.2); color: #64B5F6;">Technique A</span>
                        <h4 style="margin: 0.2rem 0; color: #F2F4F7;">Zero-Shot Prompting</h4>
                        <p style="color: #AAB5C4; font-size: 0.8rem;">Direct task instruction without exemplary demonstrations.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                res_zs = results["Zero-Shot"]
                st.caption(f"⏱️ **Latency:** {res_zs['latency']}s | 🔢 **Tokens:** ~{res_zs['tokens']}")
                with st.expander("🔍 View Raw Zero-Shot Prompt"):
                    st.code(res_zs["prompt"], language="text")
                st.markdown("##### Model Output:")
                st.markdown(res_zs["response"])

            with col_fs:
                st.markdown(
                    """
                    <div class="prompt-box">
                        <span class="badge-tag" style="background-color: rgba(76, 175, 80, 0.2); color: #81C784;">Technique B</span>
                        <h4 style="margin: 0.2rem 0; color: #F2F4F7;">Few-Shot Prompting</h4>
                        <p style="color: #AAB5C4; font-size: 0.8rem;">In-context learning with structured input-output demonstration pairs.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                res_fs = results["Few-Shot"]
                st.caption(f"⏱️ **Latency:** {res_fs['latency']}s | 🔢 **Tokens:** ~{res_fs['tokens']}")
                with st.expander("🔍 View Raw Few-Shot Prompt"):
                    st.code(res_fs["prompt"], language="text")
                st.markdown("##### Model Output:")
                st.markdown(res_fs["response"])

            with col_cot:
                st.markdown(
                    """
                    <div class="prompt-box">
                        <span class="badge-tag" style="background-color: rgba(156, 39, 176, 0.2); color: #BA68C8;">Technique C</span>
                        <h4 style="margin: 0.2rem 0; color: #F2F4F7;">Chain-of-Thought (CoT)</h4>
                        <p style="color: #AAB5C4; font-size: 0.8rem;">Forces explicit step-by-step intermediate reasoning before terminal answer.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                res_cot = results["Chain-of-Thought"]
                st.caption(f"⏱️ **Latency:** {res_cot['latency']}s | 🔢 **Tokens:** ~{res_cot['tokens']}")
                with st.expander("🔍 View Raw CoT Prompt"):
                    st.code(res_cot["prompt"], language="text")
                st.markdown("##### Model Output:")
                st.markdown(res_cot["response"])

    # -------------------------------------------------------------
    # TAB 2: Temperature & Softmax Dynamics
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 🌡️ Interactive Temperature Softmax Simulator")
        st.markdown(
            "Temperature $T$ scales the raw logits $z_i$ prior to the softmax normalization:\n"
            "$$p_i = \\frac{\\exp(z_i / T)}{\\sum_{j} \\exp(z_j / T)}$$"
        )

        c_t1, c_t2 = st.columns([1, 2])
        with c_t1:
            sim_temp = st.slider(
                "Adjust Temperature (T):",
                min_value=0.01,
                max_value=1.5,
                value=0.7,
                step=0.05,
                key="sim_temp_slider",
            )
            st.markdown(
                f"""
                <div class="neuro-card" style="margin-top: 1rem;">
                    <strong>Mathematical Regimes:</strong>
                    <ul style="color: #AAB5C4; font-size: 0.85rem; padding-left: 1.1rem; margin-top: 0.3rem;">
                        <li><strong>T &rarr; 0.0 (Greedy):</strong> Probability collapses into a Dirac delta on the argmax logit. Zero entropy.</li>
                        <li><strong>T &approx; 0.7 (Balanced):</strong> Preserves rank order while allowing natural grammatical variations.</li>
                        <li><strong>T &gt; 1.0 (High Entropy):</strong> Flattens the distribution towards uniform random sampling across the vocabulary.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Candidate next tokens and mock logits
        tokens = ["attention", "neural", "computation", "algorithm", "banana", "galaxy"]
        base_logits = [4.5, 3.8, 2.9, 2.1, 0.4, -0.8]

        res_math = PromptEngineeringLab.calculate_token_probabilities(base_logits, sim_temp)
        probs = res_math["probabilities"]
        entropy = res_math["entropy"]

        with c_t2:
            fig_temp = go.Figure(
                data=[
                    go.Bar(
                        x=tokens,
                        y=probs,
                        marker_color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                    )
                ]
            )
            fig_temp.update_layout(
                title=dict(
                    text=f"Next-Token Softmax Probability Distribution (T={sim_temp:.2f}, Shannon Entropy={entropy:.3f} bits)",
                    font=dict(color="#F2F4F7", size=13),
                ),
                xaxis=dict(title="Candidate Vocabulary Tokens", color="#AAB5C4", gridcolor="#39465A"),
                yaxis=dict(title="Probability P(w_i)", color="#AAB5C4", gridcolor="#39465A", range=[0, 1.1]),
                paper_bgcolor="#171D29",
                plot_bgcolor="#202838",
                margin=dict(l=40, r=40, t=50, b=40),
                height=350,
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Generative vs Discriminative Learning
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### ⚖️ Generative vs Discriminative Modeling")
        
        c_gd1, c_gd2 = st.columns(2)
        with c_gd1:
            st.markdown(
                """
                <div class="neuro-card">
                    <span class="badge-tag" style="background-color: rgba(76, 175, 80, 0.2); color: #81C784;">P(X, Y) or P(X)</span>
                    <h4 style="color: #81C784; margin-top: 0.2rem;">Generative Models</h4>
                    <p style="color: #AAB5C4; font-size: 0.9rem;">
                        Learn the <strong>joint probability distribution</strong> $P(X, Y)$ or data distribution $P(X)$. 
                        They capture how the data is generated and can synthesize completely new samples.
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.08);">
                    <ul style="color: #AAB5C4; font-size: 0.85rem; padding-left: 1.1rem;">
                        <li><strong>Core Formula:</strong> $P(Y|X) = \\frac{P(X, Y)}{P(X)} = \\frac{P(X|Y)P(Y)}{\\sum_y P(X|y)P(y)}$</li>
                        <li><strong>Objective:</strong> Density estimation & data generation</li>
                        <li><strong>Algorithms:</strong> Naive Bayes, GANs, VAEs, Diffusion Models, Autoregressive LLMs</li>
                        <li><strong>Application:</strong> Code generation, educational synthesis, concept artwork, molecular discovery</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_gd2:
            st.markdown(
                """
                <div class="neuro-card">
                    <span class="badge-tag" style="background-color: rgba(33, 150, 243, 0.2); color: #64B5F6;">P(Y | X)</span>
                    <h4 style="color: #64B5F6; margin-top: 0.2rem;">Discriminative Models</h4>
                    <p style="color: #AAB5C4; font-size: 0.9rem;">
                        Learn the <strong>conditional probability distribution</strong> $P(Y | X)$ directly. 
                        They optimize decision boundaries that separate classes without modeling underlying data mechanics.
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.08);">
                    <ul style="color: #AAB5C4; font-size: 0.85rem; padding-left: 1.1rem;">
                        <li><strong>Core Formula:</strong> $P(Y|X) = \\sigma(w^T X + b)$ or $f(X)$</li>
                        <li><strong>Objective:</strong> Finding optimal separating hyperplanes</li>
                        <li><strong>Algorithms:</strong> Logistic Regression, Support Vector Machines (SVM), Random Forests, ResNet</li>
                        <li><strong>Application:</strong> Spam detection, medical diagnosis, sentiment classification, face detection</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Decision Boundary vs Density Visualization
        np.random.seed(42)
        x_class0 = np.random.randn(60, 2) + np.array([-1.5, -1.5])
        x_class1 = np.random.randn(60, 2) + np.array([1.5, 1.5])

        fig_boundary = go.Figure()
        fig_boundary.add_trace(
            go.Scatter(
                x=x_class0[:, 0],
                y=x_class0[:, 1],
                mode="markers",
                name="Class A (e.g. Cat Images)",
                marker=dict(color="#2196F3", size=8),
            )
        )
        fig_boundary.add_trace(
            go.Scatter(
                x=x_class1[:, 0],
                y=x_class1[:, 1],
                mode="markers",
                name="Class B (e.g. Dog Images)",
                marker=dict(color="#FF9800", size=8),
            )
        )
        # Discriminative boundary line
        line_x = np.linspace(-3.5, 3.5, 100)
        line_y = -line_x
        fig_boundary.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Discriminative Decision Boundary [w^T x = 0]",
                line=dict(color="#E91E63", width=2, dash="dash"),
            )
        )

        fig_boundary.update_layout(
            title=dict(
                text="Discriminative Boundary (Line) vs Generative Cluster Densities (Clouds)",
                font=dict(color="#F2F4F7", size=13),
            ),
            xaxis=dict(title="Feature 1 (x_1)", color="#AAB5C4", gridcolor="#39465A"),
            yaxis=dict(title="Feature 2 (x_2)", color="#AAB5C4", gridcolor="#39465A"),
            paper_bgcolor="#171D29",
            plot_bgcolor="#202838",
            legend=dict(font=dict(color="#F2F4F7"), orientation="h", yanchor="bottom", y=1.02),
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: Generative AI Lifecycle
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### 🔄 End-to-End Generative AI Lifecycle")
        st.markdown(
            "Developing reliable generative models requires an iterative six-phase engineering pipeline:"
        )

        stages = [
            ("Phase 1: Data Collection", "Curating massive multi-terabyte web corpora, domain textbooks, papers, and code repositories."),
            ("Phase 2: Data Preparation", "Deduplication (MinHash), filtering low-quality text, PII redaction, tokenization (BPE/WordPiece)."),
            ("Phase 3: Model Architecture & Pretraining", "Self-supervised next-token prediction or masked language modeling across thousands of GPUs/TPUs."),
            ("Phase 4: Alignment & Fine-Tuning", "Instruction tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF / DPO) for helpfulness."),
            ("Phase 5: Evaluation & Safety Guardrails", "Benchmarking on MMLU, GSM8K, measuring hallucination grounding, red-teaming safety filters."),
            ("Phase 6: Deployment & Monitoring", "Quantization (AWQ/GPTQ), vLLM / TensorRT-LLM serving, drift detection, and latency tracking."),
        ]

        for title, desc in stages:
            st.markdown(
                f"""
                <div class="neuro-card" style="padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;">
                    <strong style="color: #8BAFD0; font-size: 0.95rem;">{title}</strong>
                    <div style="color: #AAB5C4; font-size: 0.85rem; margin-top: 0.2rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # TAB 5: Latent-Space Explorer
    # -------------------------------------------------------------
    with tab5:
        st.markdown("### 🌌 2D Concept Latent-Space Projection")
        st.markdown(
            "Generative representations map high-dimensional semantic concepts into dense continuous vector manifolds. "
            "Concepts with similar functional roles cluster closer together in latent space."
        )

        # Concept clusters in 2D simulated semantic space
        concepts_data = [
            {"concept": "Artificial Intelligence", "x": 0.2, "y": 0.3, "category": "General AI"},
            {"concept": "Machine Learning", "x": 0.5, "y": 0.8, "category": "General AI"},
            {"concept": "Supervised Learning", "x": 1.2, "y": 1.1, "category": "ML Paradigms"},
            {"concept": "Unsupervised Learning", "x": 1.4, "y": 0.7, "category": "ML Paradigms"},
            {"concept": "Reinforcement Learning", "x": 1.5, "y": 1.4, "category": "ML Paradigms"},
            {"concept": "Convolutional Neural Network", "x": -1.2, "y": 1.5, "category": "Deep Vision"},
            {"concept": "ResNet & ViT", "x": -1.6, "y": 1.3, "category": "Deep Vision"},
            {"concept": "Recurrent Neural Network", "x": -0.8, "y": -1.2, "category": "Sequence Models"},
            {"concept": "Transformer Attention", "x": -1.4, "y": -1.5, "category": "Sequence Models"},
            {"concept": "BERT & RoBERTa", "x": -1.8, "y": -1.3, "category": "Sequence Models"},
            {"concept": "GPT-4 & Gemini", "x": -1.5, "y": -1.8, "category": "Sequence Models"},
            {"concept": "Generative Adversarial Network", "x": 0.8, "y": -1.6, "category": "Generative Models"},
            {"concept": "Diffusion Probabilistic Models", "x": 1.2, "y": -1.4, "category": "Generative Models"},
            {"concept": "Variational Autoencoder", "x": 0.6, "y": -1.2, "category": "Generative Models"},
        ]

        categories = list(set(c["category"] for c in concepts_data))
        cat_colors = {
            "General AI": "#2196F3",
            "ML Paradigms": "#4CAF50",
            "Deep Vision": "#FF9800",
            "Sequence Models": "#9C27B0",
            "Generative Models": "#E91E63",
        }

        fig_latent = go.Figure()
        for cat in categories:
            cat_pts = [c for c in concepts_data if c["category"] == cat]
            fig_latent.add_trace(
                go.Scatter(
                    x=[p["x"] for p in cat_pts],
                    y=[p["y"] for p in cat_pts],
                    mode="markers+text",
                    name=cat,
                    text=[p["concept"] for p in cat_pts],
                    textposition="top center",
                    textfont=dict(color="#F2F4F7", size=10),
                    marker=dict(color=cat_colors[cat], size=11, line=dict(color="#F2F4F7", width=1)),
                    hovertemplate="<b>%{text}</b><br>Category: " + cat + "<br>Latent (z1, z2): (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )

        fig_latent.update_layout(
            title=dict(text="Semantic Distance & Manifold Clusters in 2D Latent Projection", font=dict(color="#F2F4F7", size=14)),
            xaxis=dict(title="Latent Dimension z_1", color="#AAB5C4", gridcolor="#39465A"),
            yaxis=dict(title="Latent Dimension z_2", color="#AAB5C4", gridcolor="#39465A"),
            paper_bgcolor="#171D29",
            plot_bgcolor="#202838",
            legend=dict(font=dict(color="#F2F4F7")),
            height=450,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_latent, use_container_width=True)
