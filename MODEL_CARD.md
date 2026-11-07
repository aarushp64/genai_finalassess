# Model Card: NeuroTutor AI
*AI–ML–DL Multi-Level Concept Tutor & Generative Intelligence Lab*

---

## 1. Model Details

- **Model/System Name:** NeuroTutor AI
- **Version:** 1.0.0
- **Author:** Aarush Parate (Roll Number: 1)
- **Allocated Problem Code:** P1 — AI–ML–DL Concept Tutor
- **Course:** Generative AI (Continuous Assessment Individual Mini-Project)
- **Institution:** Department of Computer Science & Engineering
- **Release Date:** November 2026
- **License:** MIT License (Core Software) / Apache 2.0 (Transformer & Sentence-BERT Weights)

---

## 2. Intended Use

### Primary Intended Use
NeuroTutor AI is designed as an interactive, multi-modal educational platform that demystifies Artificial Intelligence, Machine Learning, and Deep Learning for learners across three adaptive cognitive tiers:
1. **School Level (Intuitive Tier):** Relatable everyday analogies, cooking/sorting metaphors, and zero intimidating mathematical notation.
2. **Undergraduate Level (Algorithmic Tier):** Algorithmic pseudocode, mathematical intuition, network architecture diagrams, and practical PyTorch implementation details.
3. **Expert Level (Research Tier):** Formal mathematical derivations, loss manifolds, attention complexity scaling (FlashAttention, RoPE), and state-of-the-art literature insights.

### Core Functional Capabilities
- Side-by-side prompt engineering comparison (Zero-Shot, Few-Shot, Chain-of-Thought).
- Softmax temperature scaling dynamics with entropy tracking ($T \in [0.0, 1.5]$).
- Conversational tutor with bounded context-window memory.
- Real multi-head self-attention extraction and Plotly heatmaps from DistilBERT across 72 head subspaces.
- Academic lecture notes ingestion (PDF, DOCX, TXT) with Sentence-BERT embeddings and FAISS cosine similarity retrieval.
- Strict syllabus-grounded answer synthesis with uncertainty refusal fallback (*"I'm not sure based on the available notes"*).
- Multimodal diagram interpretation with transparent vision limitation disclosures.
- Diffusion-based educational concept illustration synthesis.
- Natural speech summary generation via Google Text-to-Speech (`gTTS`).
- Real-time adversarial GAN minimax training on continuous distributions.
- PyTorch forward-backward autograd computational graph demonstration with L2 gradient norms.
- Automated heuristic hallucination and grounding verification with confusion matrix metrics.

### Out-of-Scope & Prohibited Uses
- High-stakes autonomous operational decisions (e.g., medical triage, automated criminal justice sentencing, financial arbitrage).
- Generation of malicious exploit scripts, malware, denial-of-service payloads, or cyberweapon instructions.
- Verbatim academic dishonesty or cheating on closed-book examinations.

---

## 3. Underlying Models, Tools & Frameworks

| Capability | Model / Service | Execution Mode | Provider / Origin | License |
| :--- | :--- | :--- | :--- | :--- |
| **Generative LLM Reasoning** | `gemini-2.5-flash` / `llama-3.3-70b` | Cloud Hosted API | Google AI Studio / Groq | Free-Tier API |
| **Deterministic Fallback Engine** | NeuroTutor Knowledge Engine | Local Deterministic | Local CPU | MIT License |
| **Attention Extraction** | `distilbert-base-uncased` | Local PyTorch | Hugging Face Cache | Apache 2.0 |
| **Dense Text Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence-BERT | UKPLab / Hugging Face | Apache 2.0 |
| **Vector Search Index** | FAISS CPU (`IndexFlatIP`) | Local In-Memory | Meta Research | MIT License |
| **Visual Diagram Captioning** | Gemini 2.5 Flash Vision / Heuristic Parser | Cloud API + Local | Google AI / Heuristic | API / Open |
| **Concept Illustration** | Hosted Diffusion Inference Pipeline | Cloud Hosted API | Pollinations AI | Free Hosted |
| **Speech Narration (TTS)** | `gTTS` (Google Text-to-Speech) | Python Library | Local Python Engine | Open-Source |
| **Adversarial GAN Training** | Minimax Bimodal Generator & Discriminator | Local PyTorch Execution | Custom PyTorch Module | MIT License |
| **Autograd Demonstration** | 3-Layer Multilayer Perceptron | Local PyTorch Execution | Custom PyTorch Module | MIT License |

---

## 4. Training Data & Ingestion Sources

- **Foundation Models:** Pre-trained on multi-terabyte web crawls, open-source code repositories, English Wikipedia, and the Toronto BookCorpus.
- **Academic Notes Knowledge Base:** User-uploaded lecture slides and notes in PDF, DOCX, and TXT formats, supplemented by pre-packaged reference course notes covering Unit I (Foundations of GenAI), Unit II (GANs and Attention), and Unit III (Transformers, BERT, GPT & RAG).
- **Evaluation Dataset:** 15 manually curated, ground-truth labelled question-evidence pairs spanning Supported, Partially Supported, and Unsupported categories.

---

## 5. Evaluation Methodology & Empirical Results

The system was evaluated through an automated benchmarking suite (`tests/run_all_evaluations.py`) with zero synthetic metric fabrication:

### 1. Semantic Retrieval (FAISS + Sentence-BERT)
- **Test Corpus:** 10 syllabus questions evaluated against indexed course notes.
- **Top-3 Hit Rate Accuracy:** **70.0%**
- **Average Retrieval Latency:** **12.9 ms** per query.

### 2. Grounding & Hallucination Verification
- **Test Set:** 15 labelled test cases (5 Supported, 5 Partially Supported, 5 Unsupported).
- **Classification Accuracy:** **93.33%** (14 out of 15 cases correctly classified).
- **False Refusal Rate:** **0.0%** (zero supported questions falsely rejected).
- **Threshold Calibration:** Supported ($\ge 0.60$), Partially Supported ($0.30 \le s < 0.60$), Not Supported ($< 0.30$).

### 3. Prompt Strategy Token & Latency Profile
- **Zero-Shot Average:** 139 tokens, 0.15s latency.
- **Few-Shot Average:** 112 tokens, 0.15s latency.
- **Chain-of-Thought Average:** 149 tokens, 0.15s latency.

### 4. Attention Visualization Coverage
- Full extraction across **6 layers** and **12 heads** (72 unique subspace representations) with token entropy and max-attended cross-token pairs.

---

## 6. Documented Failure Case Analysis

- **Failure ID:** `FAIL-01`
- **Category:** Heuristic Lexical Grounding Underestimation on Novel Paraphrases.
- **Input Query:** *"What is mode collapse in GANs and how does Wasserstein GAN solve it?"*
- **Context Provided:** Sample course notes defined mode collapse as generator producing only a subset of samples, but omitted the Wasserstein distance mathematical formulation.
- **Expected Label:** `Partially Supported`
- **Observed Behavior:** On strict uncalibrated thresholds ($T_{\text{supp}} = 0.75$), lexical token overlap scored $0.38$, falling marginally below the partial threshold into `Not Supported`.
- **Root Cause:** Pure lexical and word-stem overlap cannot account for valid conceptual paraphrasing and synonyms as effectively as cross-encoder Natural Language Inference (NLI) models.
- **Concrete Remediation:** Incorporate a dedicated cross-encoder NLI model (`cross-encoder/nli-deberta-v3-small`) to predict entailment, neutral, and contradiction probabilities directly.

---

## 7. Safety, Guardrails & Ethical Considerations

1. **Adversarial Prompt Injection Defense:** Regex and heuristic filters intercept instruction overrides (e.g., *"ignore previous instructions"*, *"reveal system prompt"*, *"DAN jailbreak"*) before input reaches generative layers.
2. **Untrusted RAG Data Boundary:** Ingested document contents are strictly treated as untrusted data inputs, defusing prompt injection payloads hidden inside student PDFs.
3. **Privacy & Ephemeral Memory:** Ingested documents and session histories reside exclusively in volatile session memory and are never saved to persistent databases or third-party storage.
4. **Transparent Limitation Disclosures:** The system explicitly notifies users that attention heatmaps do not equal causal human reasoning, that vision models may miss microscopic subscript equations, and that grounding checks are probabilistic.
