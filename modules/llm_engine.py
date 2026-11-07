"""
NeuroTutor AI - Unified LLM Engine
Author: Aarush Parate (Roll No: 1)
Supports Google Gemini API, Groq fallback, and an intelligent offline educational fallback engine.
"""

import time
import os
import random
from typing import Dict, Any, Optional, List
import config

# Attempt imports for API providers
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class LLMEngine:
    """Unified LLM inference engine with multi-provider routing and educational offline fallback."""

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.groq_api_key = os.getenv("GROQ_API_KEY", config.GROQ_API_KEY)
        self.provider = provider

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        level: str = "Undergraduate Level",
    ) -> Dict[str, Any]:
        """
        Generate completion across available providers with latency and token tracking.
        """
        start_time = time.time()
        temperature = max(0.0, min(1.5, float(temperature)))

        # Priority 1: Google Gemini API (if key available)
        if self.api_key and GEMINI_AVAILABLE and (self.provider in ["auto", "gemini"]):
            try:
                client = genai.Client(api_key=self.api_key)
                
                config_params = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if system_instruction:
                    config_params["system_instruction"] = system_instruction
                
                # Use gemini-2.5-flash for speed and reliability
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config_params,
                )
                latency = round(time.time() - start_time, 2)
                text = response.text if hasattr(response, "text") and response.text else "No response generated."
                approx_tokens = len(text.split()) * 4 // 3

                return {
                    "text": text,
                    "provider": "Google Gemini (gemini-2.5-flash)",
                    "latency": latency,
                    "tokens_approx": approx_tokens,
                    "temperature": temperature,
                    "status": "success",
                }
            except Exception as e:
                # Fallback to Groq or local engine
                print(f"[LLMEngine] Gemini API error: {e}. Trying fallback...")

        # Priority 2: Groq API (if key available)
        if self.groq_api_key and GROQ_AVAILABLE and (self.provider in ["auto", "groq"]):
            try:
                groq_client = Groq(api_key=self.groq_api_key)
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=min(1.0, temperature),  # Groq temperature safe ceiling
                    max_tokens=max_tokens,
                )
                latency = round(time.time() - start_time, 2)
                text = completion.choices[0].message.content
                approx_tokens = len(text.split()) * 4 // 3

                return {
                    "text": text,
                    "provider": "Groq API (llama-3.3-70b-versatile)",
                    "latency": latency,
                    "tokens_approx": approx_tokens,
                    "temperature": temperature,
                    "status": "success",
                }
            except Exception as e:
                print(f"[LLMEngine] Groq API error: {e}. Falling back to Offline Engine...")

        # Priority 3: Offline High-Quality Educational Fallback Engine
        return self._generate_offline_fallback(prompt, level, temperature, start_time)

    def _generate_offline_fallback(
        self, prompt: str, level: str, temperature: float, start_time: float
    ) -> Dict[str, Any]:
        """
        Deterministic, mathematically and pedagogically sound offline fallback engine.
        Ensures zero-crash educational functionality even when API keys are absent.
        """
        prompt_lower = prompt.lower()
        time.sleep(0.15)  # Simulate minimal inference latency
        latency = round(time.time() - start_time, 2)

        # Topic detection dictionary
        topics = {
            "attention": self._get_attention_explanation(level),
            "transformer": self._get_transformer_explanation(level),
            "gan": self._get_gan_explanation(level),
            "neural network": self._get_nn_explanation(level),
            "cnn": self._get_cnn_explanation(level),
            "backpropagation": self._get_backprop_explanation(level),
            "rag": self._get_rag_explanation(level),
            "generative": self._get_gen_vs_discrim_explanation(level),
            "supervised": self._get_supervised_explanation(level),
            "unsupervised": self._get_unsupervised_explanation(level),
            "reinforcement": self._get_rl_explanation(level),
        }

        matched_content = None
        for key, content in topics.items():
            if key in prompt_lower:
                matched_content = content
                break

        if not matched_content:
            matched_content = self._get_general_ai_explanation(prompt, level)

        # Apply temperature variation to phrasing
        if temperature > 1.0:
            random_flavor = random.choice([
                "\n\n*Exploratory Note (T > 1.0):* High temperature promotes lateral associative links across cognitive representations and creative mathematical abstractions.",
                "\n\n*Probabilistic Note (T > 1.0):* At this temperature entropy, the model explores lower probability candidate tokens across its vocabulary distribution.",
            ])
            matched_content += random_flavor

        tokens_approx = len(matched_content.split()) * 4 // 3

        return {
            "text": matched_content,
            "provider": "NeuroTutor Offline AI Engine (Knowledge-Grounded Fallback)",
            "latency": latency,
            "tokens_approx": tokens_approx,
            "temperature": temperature,
            "status": "offline_fallback",
        }

    # Pedagogical response generators for the three learning levels
    def _get_attention_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 🔍 What is Attention in AI? (School Level)\n\n"
                "Imagine you are reading a long detective mystery novel. When you read the sentence:\n"
                "**'The detective picked up the key and unlocked the wooden door.'**\n\n"
                "Your brain naturally connects **'door'** with **'unlocked'** and **'key'**, while paying less attention to words like 'the' or 'and'. "
                "That's exactly what **Attention** is in AI! Instead of reading words one by one and forgetting what came earlier, the AI looks at all words at once and shines a 'mental flashlight' on the words that belong together."
                "\n\n#### 🎯 Everyday Analogy:\n"
                "- **Search Query:** You looking up 'pizza delivery near me'.\n"
                "- **Key:** The title tags on website pages.\n"
                "- **Value:** The actual pizza menu and address you get back!"
                "\n\n#### 💡 Key Takeaway:\n"
                "Attention helps computers understand context by highlighting which words matter most to each other."
            )
        elif level == "Undergraduate Level":
            return (
                "### 📐 Self-Attention Mechanism (Undergraduate Level)\n\n"
                "In traditional RNNs and LSTMs, information was processed sequentially, creating an $O(N)$ sequential bottleneck and vanishing gradient issues for long-range dependencies. "
                "The **Self-Attention** mechanism (Vaswani et al., 2017) overcomes this by computing pairwise relationships across all tokens simultaneously.\n\n"
                "#### Mathematical Formulation:\n"
                "For input embeddings $X \\in \\mathbb{R}^{N \\times d_{\\text{model}}}$, we project them into Query ($Q$), Key ($K$), and Value ($V$) matrices:\n"
                "$$Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V$$\n\n"
                "The **Scaled Dot-Product Attention** is defined as:\n"
                "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n"
                "- **Scaling factor $\\sqrt{d_k}$:** Prevents dot products from growing excessively large for high dimensions, which would push softmax gradients into regions with vanishingly small derivatives.\n"
                "- **Multi-Head Attention:** Extends this by projecting into $h$ distinct representation subspaces, enabling the model to attend to syntactic and semantic relations simultaneously."
                "\n\n#### 💻 Complexity:\n"
                "- Computational Complexity per Layer: $O(N^2 \\cdot d)$ where $N$ is sequence length.\n"
                "- Path Length: $O(1)$ maximum path length between any two token positions."
            )
        else:  # Expert Level
            return (
                "### 🔬 Multi-Head Attention & Linear Attention Scalability (Expert Level)\n\n"
                "#### Formal Formulation & Subspace Projections:\n"
                "Multi-Head Attention (MHA) maps input sequences into $h$ orthogonal representation subspaces:\n"
                "$$\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O$$\n"
                "$$\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$\n"
                "where projection matrices $W_i^Q \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}, W_i^K \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}, W_i^V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_v}, W^O \\in \\mathbb{R}^{h d_v \\times d_{\\text{model}}}$.\n\n"
                "#### Computational Bottlenecks & Hardware Optimization:\n"
                "1. **Quadratic Time and Memory ($O(N^2)$):** The attention matrix $S = QK^T / \\sqrt{d_k}$ requires materializing an $N \\times N$ tensor in GPU High Bandwidth Memory (HBM).\n"
                "2. **FlashAttention Optimization (Dao et al.):** Uses tiling to compute softmax incrementally via online softmax algorithms without materializing the full intermediate $N \\times N$ attention matrix to HBM, achieving $2\\times-4\\times$ wall-clock speedup through SRAM-bound kernel fusion.\n"
                "3. **Rotary Position Embeddings (RoPE):** Applies complex orthogonal rotation matrices $R_{\\Theta, m}^d$ directly to Query and Key vectors: $\\langle R_m q, R_n k \\rangle = g(q, k, m-n)$, preserving relative distance invariance in token manifolds."
            )

    def _get_transformer_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 🤖 What is a Transformer? (School Level)\n\n"
                "A Transformer is the brain behind modern AI like ChatGPT and Gemini. "
                "Older AI models read sentences word by word—like reading with a tiny magnifying glass. "
                "A Transformer has 'super-vision': it looks at the whole page all at once! "
                "Because it sees everything together, it understands puns, context, and long stories much better."
            )
        elif level == "Undergraduate Level":
            return (
                "### 🏗️ Transformer Architecture (Undergraduate Level)\n\n"
                "The Transformer consists of an **Encoder-Decoder** structure (or Decoder-only for GPT, Encoder-only for BERT).\n"
                "1. **Positional Encoding:** Adds sinusoidal or learned vectors $PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d})$ to preserve order without recurrence.\n"
                "2. **Multi-Head Self-Attention:** Computes pairwise token interactions.\n"
                "3. **Position-wise Feed-Forward Network (FFN):** Applies two linear transformations with a non-linearity (ReLU or GELU): $\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$.\n"
                "4. **Residual Connections & LayerNorm:** $x' = \\text{LayerNorm}(x + \\text{Sublayer}(x))$ stabilizes gradient propagation."
            )
        else:
            return (
                "### 🧬 Transformer Dynamics & Pre-Training Paradigms (Expert Level)\n\n"
                "#### Architectural Taxonomies:\n"
                "- **Encoder-only (BERT, RoBERTa):** Bidirectional masked language modeling ($p(x_m | x_{\\backslash m})$). Optimal for NLU, embeddings, and token classification.\n"
                "- **Decoder-only (GPT-4, Llama 3, Gemini):** Causal autoregressive language modeling ($p(x) = \\prod_{i=1}^N p(x_i | x_{<i})$) with lower-triangular causal attention masking.\n"
                "- **Encoder-Decoder (T5, BART):** Bidirectional representation mapped to autoregressive sequence generation.\n\n"
                "#### Training Stability & Modern Optimizations:\n"
                "- **Pre-LN vs Post-LN:** Modern models exclusively utilize Pre-Layer Normalization ($x + \\text{Sublayer}(\\text{LN}(x))$) or RMSNorm to avoid vanishing gradients during initialization without requiring prolonged warmup.\n"
                "- **SwiGLU Activations:** $\\text{SwiGLU}(x) = (xW \\cdot \\sigma(xV))W_2$ providing stronger empirical perplexity scaling over standard GELU."
            )

    def _get_gan_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 🎨 What is a GAN? (School Level)\n\n"
                "Think of a **GAN (Generative Adversarial Network)** as a game between two friends:\n"
                "1. **The Art Forger (Generator):** Tries to paint fake Mona Lisa masterpieces.\n"
                "2. **The Art Detective (Discriminator):** Tries to spot which painting is real and which is a fake.\n\n"
                "At first, the forger's paintings are silly scribbles, and the detective easily catches them. "
                "But every time the detective explains why it's a fake, the forger gets smarter! "
                "Eventually, the forger's paintings become so realistic that even the expert detective can't tell them apart!"
            )
        elif level == "Undergraduate Level":
            return (
                "### ⚔️ GAN Architecture & Minimax Formulation (Undergraduate Level)\n\n"
                "A Generative Adversarial Network (Goodfellow et al., 2014) trains two competing neural networks via a zero-sum game:\n"
                "- **Generator $G(z; \\theta_g)$:** Maps noise $z \\sim p_z(z)$ to synthetic data space $G(z) \\sim p_g$.\n"
                "- **Discriminator $D(x; \\theta_d)$:** Outputs probability $D(x) \\in [0, 1]$ that sample $x$ came from real data distribution $p_{\\text{data}}$.\n\n"
                "#### Minimax Objective Function:\n"
                "$$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}(x)}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z(z)}[\\log (1 - D(G(z)))]$$\n\n"
                "- When $D$ is optimal, the objective minimizes the **Jensen-Shannon (JS) Divergence** between $p_{\\text{data}}$ and $p_g$.\n"
                "- Global equilibrium occurs at $p_g = p_{\\text{data}}$, where $D^*(x) = \\frac{1}{2}$."
            )
        else:
            return (
                "### 🌪️ GAN Theoretical Equilibrium, Mode Collapse & WGAN (Expert Level)\n\n"
                "#### Theoretical Failures of Classical GANs:\n"
                "1. **Vanishing Gradients:** When $D$ is near-optimal early in training, $\\nabla_{\\theta_g} \\log(1 - D(G(z))) \\to 0$. In practice, using $-\\log D(G(z))$ mitigates this but induces non-zero-sum instability.\n"
                "2. **Mode Collapse:** $G$ maps diverse noise $z$ onto a restricted subset of real modes, minimizing JS divergence without covering support.\n\n"
                "#### Earth Mover's Distance & Wasserstein GAN (WGAN-GP):\n"
                "WGAN replaces JS divergence with the 1-Wasserstein metric (Earth Mover's Distance):\n"
                "$$W(p_r, p_g) = \\sup_{\\|D\\|_L \\le 1} \\mathbb{E}_{x \\sim p_r}[D(x)] - \\mathbb{E}_{y \\sim p_g}[D(y)]$$\n"
                "Enforced via Lipschitz continuity through Gradient Penalty:\n"
                "$$\\mathcal{L} = \\mathbb{E}[D(G(z))] - \\mathbb{E}[D(x)] + \\lambda \\mathbb{E}_{\\hat{x}}\\left[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2\\right]$$"
            )

    def _get_nn_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 🧠 Neural Networks (School Level)\n\n"
                "A Neural Network is like a team of helpers working in an assembly line. "
                "The first helper looks at the color of a fruit, the second helper feels its shape, and the last helper yells: 'It's an Apple!' "
                "Each helper adjusts their rules slightly every time they make a mistake, until the whole team is super accurate."
            )
        elif level == "Undergraduate Level":
            return (
                "### 🕸️ Multi-Layer Perceptron (Undergraduate Level)\n\n"
                "A feedforward neural network computes representations through stacked linear transformations and non-linear activation functions:\n"
                "$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \\quad a^{[l]} = g^{[l]}(z^{[l]})$$\n"
                "- $W^{[l]}$: Weight matrix of layer $l$.\n"
                "- $b^{[l]}$: Bias vector.\n"
                "- $g^{[l]}$: Activation function (e.g., ReLU: $g(z) = \\max(0, z)$).\n\n"
                "Trained via Backpropagation utilizing the multivariate Chain Rule to compute $\\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}}$, updated via Gradient Descent: $W^{[l]} \\leftarrow W^{[l]} - \\alpha \\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}}$."
            )
        else:
            return (
                "### ⚙️ Deep Network Optimization & Loss Manifolds (Expert Level)\n\n"
                "Universal Approximation Theorems (Cybenko, 1989; Hornik, 1991) guarantee that single hidden-layer networks can approximate any continuous function on compact subsets of $\\mathbb{R}^n$, but depth exponentially reduces parameter complexity for compositional functions.\n"
                "Key modern mechanics include Batch Normalization / LayerNorm internal covariate shift reduction, AdamW decoupled weight decay (Loshchilov & Hutter), and neural tangent kernel (NTK) analysis in overparameterized regimes."
            )

    def _get_cnn_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 👁️ Convolutional Neural Networks (School Level)\n\n"
                "A CNN is like looking through a camera with special filters. One filter looks for straight lines, another looks for circles, and another looks for colors. "
                "By combining circles and lines, the AI recognizes cat ears, whiskers, and smiles!"
            )
        elif level == "Undergraduate Level":
            return (
                "### 🖼️ CNN Feature Extraction & Pooling (Undergraduate Level)\n\n"
                "CNNs leverage two core principles:\n"
                "1. **Local Receptive Fields:** Convolution operations apply small learnable kernels across feature maps: $(I * K)(i, j) = \\sum_m \\sum_n I(i-m, j-n) K(m, n)$.\n"
                "2. **Weight Sharing:** Reduces parameter count compared to fully connected layers and provides translation equivariance.\n"
                "Followed by Pooling (Max/Average) for spatial downsampling and translation invariance."
            )
        else:
            return (
                "### 🔬 Modern Visual Backbones & ConvNeXt vs ViT (Expert Level)\n\n"
                "Modern CNN architectures (ResNet, ConvNeXt) utilize residual skip connections $y = \\mathcal{F}(x, \\{W_i\\}) + x$ to preserve identity gradients throughout hundreds of layers. "
                "Unlike Vision Transformers (ViT) which lack inductive bias and require massive pretraining corpora ($JFT-300M$), CNNs retain strong spatial locality priors and linear inference efficiency for dense downstream prediction tasks."
            )

    def _get_backprop_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 🔄 Backpropagation (School Level)\n\n"
                "Backpropagation is how an AI learns from its mistakes! "
                "Imagine shooting a basketball. If your shot lands 2 feet to the left, your brain rewinds what your hand did and adjusts your elbow. "
                "Backpropagation is the AI rewinding its steps and nudging its mathematical dials so next time the shot goes straight into the hoop!"
            )
        elif level == "Undergraduate Level":
            return (
                "### 📐 Backpropagation Algorithm (Undergraduate Level)\n\n"
                "Backpropagation computes the analytical gradient of scalar loss $\\mathcal{L}$ with respect to all network weights using reverse-mode automatic differentiation:\n"
                "$$\\delta^{[l]} = \\frac{\\partial \\mathcal{L}}{\\partial z^{[l]}} = \\left( (W^{[l+1]})^T \\delta^{[l+1]} \\right) \\odot g'^{[l]}(z^{[l]})$$\n"
                "$$\\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}} = \\delta^{[l]} (a^{[l-1]})^T, \\quad \\frac{\\partial \\mathcal{L}}{\\partial b^{[l]}} = \\delta^{[l]}$$\n"
                "Allows $O(E)$ time computation of all gradients, where $E$ is the number of edges in the computational graph."
            )
        else:
            return (
                "### ⚡ Automatic Differentiation & Computational Graph Optimization (Expert Level)\n\n"
                "Backpropagation in modern frameworks (PyTorch autograd) implements reverse-mode automatic differentiation (Reverse AD) over dynamically constructed Directed Acyclic Graphs (DAGs). "
                "Gradient checkpointing trades $O(N)$ memory for $O(N)$ recomputation by storing only boundary activations during forward pass, allowing training of deep architectures under strict GPU VRAM budgets."
            )

    def _get_rag_explanation(self, level: str) -> str:
        if level == "School Level":
            return (
                "### 📚 Retrieval-Augmented Generation (RAG) (School Level)\n\n"
                "Imagine you are taking an open-book exam. "
                "Instead of memorizing the entire encyclopedia, you flip to the exact chapter in your textbook, read the facts, and then write down your answer. "
                "**RAG** does this for AI! It looks up your uploaded documents first, grabs the exact facts, and uses them to answer your questions accurately without guessing."
            )
        elif level == "Undergraduate Level":
            return (
                "### 🔍 Retrieval-Augmented Generation Architecture (Undergraduate Level)\n\n"
                "RAG enhances LLM generations by retrieving grounding evidence from an external vectorized knowledge corpus:\n"
                "1. **Chunking & Embedding:** Documents are partitioned into chunks ($C_i$) and encoded via Sentence-BERT into dense vectors: $v_i = f_{\\text{enc}}(C_i) \\in \\mathbb{R}^d$.\n"
                "2. **Indexing:** Vectors are indexed in FAISS using Inner Product or Euclidean distance.\n"
                "3. **Retrieval:** User query $q$ is embedded ($v_q$) and top-$k$ nearest chunks are retrieved: $\\text{top}_k = \\text{argmax}_{i} (v_q \\cdot v_i)$.\n"
                "4. **Synthesis:** Context is passed into the prompt template: `Prompt = 'Context: ' + Retrieved_Chunks + ' Question: ' + q`."
            )
        else:
            return (
                "### 🎯 Advanced RAG Paradigms & Grounding Verification (Expert Level)\n\n"
                "Modern production RAG pipelines address standard bi-encoder retrieval limitations through:\n"
                "- **Hybrid Retrieval:** Dense embeddings (HNSW/FAISS) combined with sparse BM25/SPLADE via Reciprocal Rank Fusion (RRF).\n"
                "- **Cross-Encoder Re-Ranking:** Top-50 candidates are rescored with a cross-encoder model ($p(\\text{relevant} | q, c)$) to capture token-level interaction.\n"
                "- **Self-RAG & Hallucination Mitigation:** Models generate reflection tokens (`[RETRIEVE]`, `[IS_SUPPORTED]`) to verify claim entailment against the retrieved context manifold."
            )

    def _get_gen_vs_discrim_explanation(self, level: str) -> str:
        return (
            "### ⚖️ Generative vs Discriminative AI\n\n"
            "- **Generative Models:** Model the joint probability distribution $P(X, Y)$ or data distribution $P(X)$. They learn how data is generated and can produce novel synthetic instances. *Examples:* GANs, VAEs, Diffusion Models, LLMs.\n"
            "- **Discriminative Models:** Model the conditional probability $P(Y | X)$ directly. They learn decision boundaries to separate classes. *Examples:* Logistic Regression, SVM, Random Forest, ResNet classification.\n\n"
            "$$\\text{Generative: } P(Y|X) = \\frac{P(X, Y)}{P(X)} = \\frac{P(X|Y)P(Y)}{\\sum_y P(X|y)P(y)} \\quad \\text{vs} \\quad \\text{Discriminative: } P(Y|X) = f(X)$$"
        )

    def _get_supervised_explanation(self, level: str) -> str:
        return "### 🏷️ Supervised Learning\n\nSupervised learning learns a mapping function $f: X \\to Y$ from labelled pairs $(x_i, y_i)$. Covers regression (continuous outputs) and classification (discrete categories)."

    def _get_unsupervised_explanation(self, level: str) -> str:
        return "### 🧩 Unsupervised Learning\n\nUnsupervised learning discovers latent patterns, groupings, and low-dimensional manifolds in unlabelled data $X$ without explicit targets. Covers K-Means, PCA, Autoencoders, and Density Estimation."

    def _get_rl_explanation(self, level: str) -> str:
        return "### 🎮 Reinforcement Learning\n\nRL optimizes an agent's policy $\\pi(a|s)$ to maximize expected cumulative discounted rewards $R = \\sum_t \\gamma^t r_t$ through trial-and-error interactions modeled as a Markov Decision Process (MDP)."

    def _get_general_ai_explanation(self, prompt: str, level: str) -> str:
        return (
            f"### 🤖 NeuroTutor AI Explanation: {prompt.title()}\n\n"
            f"**Level:** {level}\n\n"
            "Artificial Intelligence represents the computational paradigm of creating systems capable of performing cognitive tasks that traditionally require human intelligence. "
            "Machine Learning represents the statistical subset that learns representations from empirical data, while Deep Learning leverages multi-layered hierarchical neural networks to automatically extract features from high-dimensional spaces.\n\n"
            "#### Core Principles:\n"
            "- **Representation Learning:** Mapping raw data to continuous vector spaces.\n"
            "- **Loss Optimization:** Minimizing empirical risk over parameter distributions via gradient descent.\n"
            "- **Generalization:** Ensuring low test error on out-of-distribution or unseen inputs."
        )
