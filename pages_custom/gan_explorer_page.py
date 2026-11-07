"""
NeuroTutor AI - GAN Explorer & PyTorch Implementation Page
Author: Aarush Parate (Roll No: 1)
Unit II (GANs) & Unit IV (PyTorch Neural Network & Autograd Demonstration)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.gan_explorer import GANExplorerEngine
from modules.pytorch_demo import PyTorchDemoEngine


def render_gan_explorer():
    """Renders the GAN Explorer and PyTorch Deep Learning demonstration page."""
    st.markdown(
        """
        <div class="neuro-card-accent">
            <span class="badge-tag">Units II & IV: GANs & PyTorch Implementation</span>
            <h2 style="margin: 0.2rem 0; color: #F2F2F2;">GAN Explorer & PyTorch Implementation Lab</h2>
            <p style="color: #AEB7BF; font-size: 0.95rem; margin-top: 0.3rem;">
                Train an interactive PyTorch GAN in real time to capture continuous distributions, 
                and inspect dynamic forward-backward computational graphs in an educational neural network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_gan, tab_pytorch = st.tabs(["⚔️ PyTorch GAN Adversarial Explorer", "🧠 PyTorch Neural Network & Autograd Demo"])

    # -------------------------------------------------------------
    # TAB 1: PyTorch GAN Explorer
    # -------------------------------------------------------------
    with tab_gan:
        st.markdown("### 🎨 Real-Time PyTorch GAN Training")
        st.markdown(
            "Observe the zero-sum minimax competition between the Generator $G(z)$ and Discriminator $D(x)$:\n"
            "$$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]$$\n"
            "The ground-truth target is a bimodal Gaussian mixture: $p(x) = 0.5 \\cdot \\mathcal{N}(-2.5, 0.5^2) + 0.5 \\cdot \\mathcal{N}(2.5, 0.5^2)$."
        )

        # Initialize or retrieve GAN engine in session state
        if "gan_engine" not in st.session_state:
            st.session_state["gan_engine"] = GANExplorerEngine(latent_dim=4)

        gan = st.session_state["gan_engine"]

        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1, 1.5])
        with c_ctrl1:
            epochs_to_train = st.slider("Epochs per Click:", min_value=10, max_value=100, value=25, step=5)
        with c_ctrl2:
            st.write("")
            train_btn = st.button("⚡ Train GAN", use_container_width=True)
        with c_ctrl3:
            st.write("")
            reset_btn = st.button("🔄 Reset Weights", use_container_width=True)

        if reset_btn:
            gan.reset_models()
            st.success("GAN models and training history reset to Epoch 0.")
            st.rerun()

        if train_btn:
            with st.spinner(f"Training Generator & Discriminator for {epochs_to_train} epochs..."):
                metrics = gan.train_steps(num_epochs=epochs_to_train)
                st.session_state["latest_gan_metrics"] = metrics

        # Visualizations
        col_v1, col_v2 = st.columns([3, 2])
        with col_v1:
            fig_dist = gan.generate_distribution_plot(num_samples=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_v2:
            fig_loss = gan.generate_loss_curve()
            st.plotly_chart(fig_loss, use_container_width=True)

        # Architectural Educational Card
        st.markdown("#### 📐 GAN Network Architecture (PyTorch):")
        c_arch1, c_arch2 = st.columns(2)
        with c_arch1:
            st.markdown(
                r"""
                <div class="neuro-card">
                    <h5 style="color: #FF9800; margin-top: 0;">Generator Network G(z)</h5>
                    <ul style="color: #AEB7BF; font-size: 0.85rem; padding-left: 1.1rem; line-height: 1.5;">
                        <li><strong>Input:</strong> Latent vector $z \sim \mathcal{N}(0, I_4)$</li>
                        <li><code>Linear(4 &rarr; 32) + ReLU</code></li>
                        <li><code>Linear(32 &rarr; 32) + ReLU</code></li>
                        <li><code>Linear(32 &rarr; 1)</code> (Continuous Output)</li>
                        <li><strong>Loss:</strong> $\mathcal{L}_G = -\mathbb{E}_z[\log D(G(z))]$</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_arch2:
            st.markdown(
                r"""
                <div class="neuro-card">
                    <h5 style="color: #2196F3; margin-top: 0;">Discriminator Network D(x)</h5>
                    <ul style="color: #AEB7BF; font-size: 0.85rem; padding-left: 1.1rem; line-height: 1.5;">
                        <li><strong>Input:</strong> Sample $x \in \mathbb{R}^1$ (Real or Synthetic)</li>
                        <li><code>Linear(1 &rarr; 32) + LeakyReLU(0.2)</code></li>
                        <li><code>Linear(32 &rarr; 32) + LeakyReLU(0.2)</code></li>
                        <li><code>Linear(32 &rarr; 1) + Sigmoid</code></li>
                        <li><strong>Loss:</strong> $\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]$</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # TAB 2: PyTorch Neural Network & Autograd Demo
    # -------------------------------------------------------------
    with tab_pytorch:
        st.markdown("### 🧠 PyTorch Neural Network & Backward Autograd Demo")
        st.markdown(
            r"An educational PyTorch multi-layer perceptron running locally on CPU. "
            r"Inspect raw tensor transformations, activation layers, cross-entropy loss, and analytical gradient backpropagation ($\partial \mathcal{L} / \partial W$)."
        )

        if "pytorch_demo_engine" not in st.session_state:
            st.session_state["pytorch_demo_engine"] = PyTorchDemoEngine()

        pt_engine = st.session_state["pytorch_demo_engine"]

        c_f1, c_f2, c_f3, c_f4, c_target = st.columns(5)
        with c_f1:
            x1 = st.number_input("Feature x_1:", value=1.5, step=0.1)
        with c_f2:
            x2 = st.number_input("Feature x_2:", value=-0.8, step=0.1)
        with c_f3:
            x3 = st.number_input("Feature x_3:", value=2.2, step=0.1)
        with c_f4:
            x4 = st.number_input("Feature x_4:", value=0.5, step=0.1)
        with c_target:
            target_cls = st.selectbox("Target Label y:", options=[0, 1], index=1)

        if st.button("🚀 Execute Forward & Backward Step", use_container_width=True):
            step_res = pt_engine.run_step_demo([x1, x2, x3, x4], target_class=target_cls)
            st.session_state["latest_pytorch_step"] = step_res

        if "latest_pytorch_step" in st.session_state:
            res = st.session_state["latest_pytorch_step"]

            m_l1, m_l2, m_l3 = st.columns(3)
            with m_l1:
                st.metric("Cross-Entropy Loss (L)", f"{res['loss_value']:.5f}")
            with m_l2:
                st.metric("Predicted Class", res["predicted_class"], delta=f"Target: {target_cls}")
            with m_l3:
                probs = res["class_probabilities"]
                st.metric("Softmax Probabilities", f"[{probs[0]:.3f}, {probs[1]:.3f}]")

            st.write("")
            fig_grad = PyTorchDemoEngine.create_gradient_flow_chart(res["gradient_norms"])
            st.plotly_chart(fig_grad, use_container_width=True)

            with st.expander("🔍 Inspect Internal Layer Activations & Shapes:"):
                st.json({
                    "Activation Shapes": res["activation_shapes"],
                    "Layer 1 Post-ReLU Activations (A1)": [round(v, 4) for v in res["layer_1_activations"]],
                    "Layer 2 Post-ReLU Activations (A2)": [round(v, 4) for v in res["layer_2_activations"]],
                    "Weight Tensor Norms (L2)": res["weight_norms"],
                    "Gradient Norms (dL/dW)": res["gradient_norms"],
                })
