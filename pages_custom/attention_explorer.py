"""
NeuroTutor AI - Attention Explorer Page
Author: Aarush Parate (Roll No: 1)
Unit II: GANs and Large Language Models
Real multi-head attention weight extraction using Hugging Face DistilBERT and interactive Plotly heatmaps.
"""

import streamlit as st
import numpy as np
from modules.attention_visualizer import AttentionVisualizer
from modules.safety_guardrails import SafetyGuardrails


def render_attention_explorer(visualizer: AttentionVisualizer):
    """Renders the real Transformer Attention Explorer page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit II: Transformer Attention</span>
            <h2 style="margin: 0.2rem 0; color: #F2F2F2;">Transformer Attention Explorer</h2>
            <p style="color: #AEB7BF; font-size: 0.95rem; margin-top: 0.3rem;">
                Extract and inspect <strong>real attention weight matrices</strong> from a local Hugging Face DistilBERT transformer 
                across 6 layers and 12 multi-head attention subspaces.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input text for attention inspection
    with st.form("attention_form"):
        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            attn_text = st.text_input(
                "Enter text for attention analysis:",
                value="Deep learning neural networks utilize self attention mechanisms.",
                help="Sentence to pass through DistilBERT to extract attention matrices.",
            )
        with col_btn:
            st.write("")
            submitted = st.form_submit_button("🔍 Extract Attention", use_container_width=True)

    if submitted or "current_attention_data" not in st.session_state:
        is_safe, msg = SafetyGuardrails.validate_user_prompt(attn_text)
        if not is_safe:
            st.error(f"🛡️ Safety Alert: {msg}")
            return

        with st.spinner("Tokenizing text and computing attention weights through DistilBERT..."):
            data = visualizer.extract_attentions(attn_text)
            st.session_state["current_attention_data"] = data
            st.session_state["attention_input_text"] = attn_text

    data = st.session_state.get("current_attention_data")
    if not data:
        st.info("Please submit a sentence above to extract attention weights.")
        return

    tokens = data["tokens"]
    num_layers = data["num_layers"]
    num_heads = data["num_heads"]
    attentions_by_layer = data["attentions_by_layer"]

    # Controls: Layer and Head selector
    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        selected_layer = st.selectbox(
            "Select Transformer Layer:",
            options=list(range(num_layers)),
            format_func=lambda x: f"Layer {x + 1} (of {num_layers})",
            index=num_layers - 1,  # Default to last layer where high-level semantic relations concentrate
        )
    with c_ctrl2:
        head_options = ["Average Across All Heads"] + [f"Head {h + 1}" for h in range(num_heads)]
        selected_head_str = st.selectbox(
            "Select Attention Head:",
            options=head_options,
            index=0,
        )
        selected_head = None if selected_head_str == "Average Across All Heads" else int(selected_head_str.split()[1]) - 1

    # Extract selected layer matrix: shape (num_heads, seq_len, seq_len)
    layer_matrix = attentions_by_layer[selected_layer]

    # Generate Plotly Heatmap
    fig_heatmap = visualizer.create_plotly_heatmap(
        tokens=tokens,
        attn_matrix=layer_matrix,
        layer_idx=selected_layer,
        head_idx=selected_head,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Attention Head Metrics
    if selected_head is not None:
        head_matrix = layer_matrix[selected_head]
    else:
        head_matrix = np.mean(layer_matrix, axis=0)

    # Find highest non-diagonal token pair
    seq_len = len(tokens)
    np_copy = head_matrix.copy()
    np.fill_diagonal(np_copy, 0.0)  # Suppress self-loops to highlight cross-token relations
    max_idx = np.unravel_index(np.argmax(np_copy), np_copy.shape)
    q_tok = tokens[max_idx[0]]
    k_tok = tokens[max_idx[1]]
    max_val = np_copy[max_idx]

    # Calculate mean entropy across rows
    entropies = []
    for row in head_matrix:
        row_clean = row[row > 1e-9]
        if len(row_clean) > 0:
            entropies.append(-np.sum(row_clean * np.log2(row_clean)))
    mean_entropy = float(np.mean(entropies))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-val">{len(tokens)}</div>
                <div class="stat-label">Sequence Tokens</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-val" style="font-size: 1.25rem;">'{q_tok}' &rarr; '{k_tok}'</div>
                <div class="stat-label">Max Cross-Token Attention ({max_val:.3f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-val">{mean_entropy:.3f} bits</div>
                <div class="stat-label">Head Attention Entropy (Focus)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Academic & Interpretability Caveats
    st.markdown(
        """
        <div class="neuro-card" style="border-left: 4px solid #FF9800;">
            <h4 style="color: #FFE082; margin-top: 0;">⚠️ Academic Interpretation Notice: Attention &ne; Explanation</h4>
            <p style="color: #AEB7BF; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0.4rem;">
                In peer-reviewed literature (e.g., <em>Jain & Wallace, 2019 "Attention is not Explanation"</em>; 
                <em>Wiegreffe & Pinter, 2019 "Attention is not not Explanation"</em>), attention weights reflect the mathematical 
                routing of vectors between representation subspaces during forward inference.
            </p>
            <ul style="color: #AEB7BF; font-size: 0.85rem; padding-left: 1.2rem; margin: 0;">
                <li>Attention weights do not necessarily represent human-like causal rationale.</li>
                <li>Multiple distinct attention patterns can yield mathematically identical model outputs.</li>
                <li>Different heads naturally specialize: early heads often track syntax and neighboring bigrams, while deeper heads capture long-range semantic anaphora.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
