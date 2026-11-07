"""
NeuroTutor AI - Knowledge Base & RAG Page
Author: Aarush Parate (Roll No: 1)
Unit III: GPT, BERT and Multimodal Models
Academic notes ingestion, Sentence-BERT (all-MiniLM-L6-v2) embedding, FAISS semantic retrieval, grounded answers, and GPT vs BERT comparison.
"""

import streamlit as st
from modules.rag_engine import RAGEngine
from modules.document_processor import DocumentProcessor
from modules.llm_engine import LLMEngine
from modules.hallucination_checker import HallucinationChecker
from modules.safety_guardrails import SafetyGuardrails
import config


def render_knowledge_base(rag: RAGEngine, llm: LLMEngine):
    """Renders the Knowledge Base and Retrieval-Augmented Generation page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Unit III: Knowledge Base & RAG</span>
            <h2 style="margin: 0.2rem 0; color: #F2F4F7;">Academic Knowledge Base & Semantic Search</h2>
            <p style="color: #AAB5C4; font-size: 0.95rem; margin-top: 0.3rem;">
                Upload academic notes (PDF, DOCX, TXT), index them using <strong>Sentence-BERT (all-MiniLM-L6-v2)</strong> and <strong>FAISS</strong>, 
                and receive strict syllabus-grounded answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_rag, tab_gpt_bert = st.tabs(["📚 Notes Ingestion & RAG Query", "⚖️ GPT vs BERT Architectural Comparison"])

    with tab_rag:
        # Document Upload Section
        st.markdown("### 📥 1. Ingest Course Materials")
        c_up1, c_up2 = st.columns([3, 2])

        with c_up1:
            uploaded_files = st.file_uploader(
                "Upload Lecture Notes or Papers (PDF, DOCX, TXT):",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )

        with c_up2:
            st.markdown(
                """
                <div class="neuro-card" style="padding: 0.8rem 1rem;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #F2F4F7;">Pre-loaded Sample Notes Available:</div>
                    <div style="font-size: 0.8rem; color: #AAB5C4; margin-top: 0.2rem;">
                        Foundations of GenAI, Deep Learning Architectures, and Transformers & RAG course notes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            load_samples_btn = st.button("📂 Load Pre-Packaged Course Notes", use_container_width=True)

        # Handle loading pre-packaged notes
        if load_samples_btn:
            with st.spinner("Indexing pre-packaged academic notes via Sentence-BERT..."):
                sample_files = list(config.SAMPLE_NOTES_DIR.glob("*.txt"))
                all_chunks = []
                for sf in sample_files:
                    raw_text = sf.read_text(encoding="utf-8")
                    chunks = DocumentProcessor.chunk_text(raw_text, source_name=sf.name)
                    all_chunks.extend(chunks)

                count = rag.index_chunks(all_chunks)
                st.session_state["indexed_chunk_count"] = rag.index.ntotal
                st.success(f"Successfully indexed {count} semantic chunks into FAISS!")

        # Handle user uploads
        if uploaded_files:
            new_chunks = []
            for uf in uploaded_files:
                file_bytes = uf.read()
                is_valid, v_msg = SafetyGuardrails.validate_file_upload(uf.name, len(file_bytes))
                if not is_valid:
                    st.error(f"File Rejected '{uf.name}': {v_msg}")
                    continue

                try:
                    text = DocumentProcessor.extract_text_from_bytes(file_bytes, uf.name)
                    chunks = DocumentProcessor.chunk_text(text, source_name=uf.name)
                    new_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Error parsing '{uf.name}': {str(e)}")

            if new_chunks:
                with st.spinner(f"Encoding {len(new_chunks)} passages using Sentence-BERT..."):
                    rag.index_chunks(new_chunks)
                    st.session_state["indexed_chunk_count"] = rag.index.ntotal
                    st.success(f"Indexed {len(new_chunks)} passages from uploaded files!")

        # Vector Index Status Banner
        total_indexed = rag.index.ntotal
        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.metric("Total Indexed Passages", total_indexed)
        with c_stat2:
            st.metric("Embedding Model", config.EMBEDDING_MODEL_NAME)
        with c_stat3:
            st.metric("Embedding Dimension", f"{rag.dimension} dense vectors")

        if total_indexed == 0:
            st.info("💡 Tip: Click **'Load Pre-Packaged Course Notes'** above to test semantic retrieval immediately!")

        st.write("")
        st.markdown("---")

        # Grounded Query Interface
        st.markdown("### 🔍 2. Ask Questions Based on Ingested Notes")
        with st.form("rag_query_form"):
            user_query = st.text_input(
                "Enter your question:",
                placeholder="e.g. What is the formula for scaled dot-product attention? How does BERT differ from GPT?",
            )
            col_q1, col_q2 = st.columns([1, 4])
            with col_q1:
                query_submitted = st.form_submit_button("🔎 Retrieve & Synthesize", use_container_width=True)

        if query_submitted:
            if not user_query.strip():
                st.warning("Please type a question to search your notes.")
                return

            is_safe, msg = SafetyGuardrails.validate_user_prompt(user_query)
            if not is_safe:
                st.error(f"🛡️ Safety Alert: {msg}")
                return

            if total_indexed == 0:
                st.warning("No notes indexed yet. Please upload notes or load pre-packaged course materials first.")
                return

            with st.spinner("Searching FAISS vector index and synthesizing grounded explanation..."):
                rag_result = rag.query_with_grounding(
                    query=user_query,
                    llm=llm,
                    level="Undergraduate Level",
                )

                st.session_state["latest_rag_result"] = rag_result
                st.session_state["latest_rag_query"] = user_query

        # Display Grounded Answer & Retrieved Chunks
        if "latest_rag_result" in st.session_state:
            result = st.session_state["latest_rag_result"]
            answer = result["answer"]
            passages = result["retrieved_passages"]

            # Run Grounding Checker
            grounding = HallucinationChecker.check_grounding(answer, passages)
            status = grounding["status"]
            score = grounding["grounding_score"]

            st.write("")
            st.markdown("#### 🎯 Grounded AI Response")

            # Grounding Status Banner
            if status == "Supported":
                st.markdown(
                    f"""
                    <div class="grounding-supported">
                        <strong>🟢 Grounding Status: Supported ({score*100:.1f}%)</strong><br>
                        <span style="font-size: 0.85rem;">Claims are strongly corroborated by retrieved academic notes.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif status == "Partially Supported":
                st.markdown(
                    f"""
                    <div class="grounding-partial">
                        <strong>🟡 Grounding Status: Partially Supported ({score*100:.1f}%)</strong><br>
                        <span style="font-size: 0.85rem;">Some statements rely on general AI knowledge outside the indexed text.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="grounding-unsupported">
                        <strong>🔴 Grounding Status: Not Supported / Uncertainty Refusal</strong><br>
                        <span style="font-size: 0.85rem;">The available notes lack sufficient evidence for this query.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="neuro-card">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Retrieved Evidence Inspection Accordion
            if passages:
                st.markdown("##### 📄 Retrieved Evidence Passages (FAISS Top Chunks):")
                for idx, p in enumerate(passages):
                    with st.expander(
                        f"Chunk #{p['chunk_id']} from '{p['source']}' — Similarity Score: {p['similarity_score']:.4f}"
                    ):
                        st.write(p["text"])

    # -------------------------------------------------------------
    # TAB 2: GPT vs BERT Comparison
    # -------------------------------------------------------------
    with tab_gpt_bert:
        st.markdown("### ⚖️ Architectural Comparison: GPT vs. BERT")
        st.markdown(
            "Transformers branch into two fundamental architectural paradigms based on attention masking and pre-training objectives:"
        )

        c_gpt, c_bert = st.columns(2)
        with c_gpt:
            st.markdown(
                r"""
                <div class="neuro-card" style="border-top: 3px solid #2196F3;">
                    <span class="badge-tag" style="background-color: rgba(33, 150, 243, 0.2); color: #64B5F6;">Decoder-Only</span>
                    <h3 style="color: #64B5F6; margin-top: 0.2rem;">GPT Family (Generative Pre-trained Transformer)</h3>
                    <p style="color: #AAB5C4; font-size: 0.9rem;">
                        Optimized for <strong>autoregressive sequence generation</strong>.
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.08);">
                    <ul style="color: #AAB5C4; font-size: 0.85rem; padding-left: 1.1rem; line-height: 1.6;">
                        <li><strong>Attention Mask:</strong> Causal lower-triangular mask. Tokens can only attend to past positions: $j \le i$.</li>
                        <li><strong>Pre-training Objective:</strong> Next-Token Prediction: $\max_\theta \sum_t \log P(x_t | x_{<t}; \theta)$.</li>
                        <li><strong>Directionality:</strong> Unidirectional (Left-to-Right).</li>
                        <li><strong>Primary Strengths:</strong> Creative writing, conversational agents, code generation, in-context zero/few-shot learning.</li>
                        <li><strong>Notable Models:</strong> GPT-4, Llama 3, Gemini, Mistral.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_bert:
            st.markdown(
                """
                <div class="neuro-card" style="border-top: 3px solid #4CAF50;">
                    <span class="badge-tag" style="background-color: rgba(76, 175, 80, 0.2); color: #81C784;">Encoder-Only</span>
                    <h3 style="color: #81C784; margin-top: 0.2rem;">BERT Family (Bidirectional Encoder)</h3>
                    <p style="color: #AAB5C4; font-size: 0.9rem;">
                        Optimized for <strong>dense semantic representation & language understanding</strong>.
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.08);">
                    <ul style="color: #AAB5C4; font-size: 0.85rem; padding-left: 1.1rem; line-height: 1.6;">
                        <li><strong>Attention Mask:</strong> Unconstrained bidirectional mask (all ones). Every token attends to every other token.</li>
                        <li><strong>Pre-training Objective:</strong> Masked Language Modeling (MLM: 15% tokens masked) + Next Sentence Prediction (NSP).</li>
                        <li><strong>Directionality:</strong> Fully bidirectional.</li>
                        <li><strong>Primary Strengths:</strong> Semantic search embeddings, sentiment analysis, NER, extractive QA.</li>
                        <li><strong>Notable Models:</strong> BERT, RoBERTa, DeBERTa, Sentence-BERT.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Attention Mask Matrix Comparison:")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(r"**GPT Causal Lower-Triangular Mask ($M_{ij} = -\infty$ for $j > i$):**")
            st.code(
                "[[ 1,  0,  0,  0 ],\n"
                " [ 1,  1,  0,  0 ],\n"
                " [ 1,  1,  1,  0 ],\n"
                " [ 1,  1,  1,  1 ]]",
                language="python",
            )
        with c_m2:
            st.markdown("**BERT Full Bidirectional Mask (Every token attends to all positions):**")
            st.code(
                "[[ 1,  1,  1,  1 ],\n"
                " [ 1,  1,  1,  1 ],\n"
                " [ 1,  1,  1,  1 ],\n"
                " [ 1,  1,  1,  1 ]]",
                language="python",
            )
