"""
Capture 13 high-resolution, captioned screenshots from the live NeuroTutor AI application
for inclusion in the formal 16-page academic report.
"""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE_DIR / "assets" / "report_screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"


def wait_for_idle(page, timeout=30000):
    try:
        page.wait_for_selector('[data-testid="stSkeleton"]', state="detached", timeout=timeout)
    except Exception:
        pass
    try:
        page.wait_for_selector('.stSpinner', state="detached", timeout=timeout)
    except Exception:
        pass
    page.wait_for_timeout(800)


def click_nav(page, nav_text):
    wait_for_idle(page)
    target = page.locator(f'div[role="radiogroup"] label:has-text("{nav_text}")').first
    target.scroll_into_view_if_needed()
    target.click()
    page.wait_for_timeout(1000)
    wait_for_idle(page)


def capture_all():
    print("Starting screenshot capture from:", APP_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 768})

        # Figure 2: Dashboard Hero & Top Section
        print("Capturing Figure 2 (Dashboard Hero)...")
        page.goto(APP_URL, wait_until="networkidle", timeout=35000)
        wait_for_idle(page, timeout=45000)
        page.wait_for_selector("text=NeuroTutor AI", timeout=20000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig2_dashboard_hero.png"))

        # Figure 3: Dashboard Learning Levels & Syllabus Grid
        print("Capturing Figure 3 (Dashboard Levels & Units)...")
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig3_dashboard_levels.png"))

        # Figure 4: AI Concept Tutor - Question & Response
        print("Capturing Figure 4 (Concept Tutor)...")
        click_nav(page, "AI Concept Tutor")
        q_inp = page.locator('input[placeholder="Type your question here..."]').first
        q_inp.fill("Explain how a Transformer neural network works, including self-attention, positional encoding, and feed-forward layers.")
        page.locator('button:has-text("Generate Answer")').first.click()
        wait_for_idle(page)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 350)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig4_concept_tutor.png"))

        # Figure 5: Spoken Audio Synthesis
        print("Capturing Figure 5 (Audio Synthesis)...")
        aud_btn = page.locator('button:has-text("Synthesize Audio")').first
        if aud_btn.is_visible():
            aud_btn.click()
            wait_for_idle(page)
            page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 350)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig5_audio_synthesis.png"))

        # Figure 6: Prompt Engineering Lab Triple Comparison
        print("Capturing Figure 6 (Prompt Lab Comparison)...")
        click_nav(page, "Prompt Engineering Lab")
        c_inp = page.locator('input[aria-label="Enter Concept to Compare:"]').first
        c_inp.fill("Explain the difference between Artificial Intelligence, Machine Learning, and Deep Learning.")
        page.locator('button:has-text("Run Triple-Technique Comparison")').first.click()
        wait_for_idle(page, timeout=35000)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 350)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig6_prompt_comparison.png"))

        # Figure 7: Temperature Softmax Dynamics Plot
        print("Capturing Figure 7 (Softmax Temperature Dynamics)...")
        page.locator('button[role="tab"]:has-text("Temperature & Softmax Dynamics")').first.click()
        page.wait_for_timeout(1000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig7_softmax_dynamics.png"))

        # Figure 8: DistilBERT Attention Heatmap
        print("Capturing Figure 8 (Attention Heatmap)...")
        click_nav(page, "Attention Explorer")
        attn_inp = page.locator('input[aria-label="Enter text for attention analysis:"]').first
        attn_inp.fill("The student studied machine learning because the examination was difficult.")
        page.locator('button:has-text("Extract Attention")').first.click()
        wait_for_idle(page)
        page.wait_for_selector(".js-plotly-plot", timeout=20000)
        page.wait_for_timeout(2500)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig8_attention_heatmap.png"))

        # Figure 9: Knowledge Base Notes Ingestion & Index Status
        print("Capturing Figure 9 (Notes Ingestion & Status)...")
        click_nav(page, "Knowledge Base & RAG")
        load_btn = page.locator('button:has-text("Load Pre-Packaged Course Notes")').first
        if load_btn.is_visible():
            load_btn.click()
            wait_for_idle(page)
            page.wait_for_timeout(1500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig9_rag_indexing.png"))

        # Figure 10: Grounded AI Response & Evidence
        print("Capturing Figure 10 (Grounded QA & Evidence)...")
        rag_inp = page.locator('input[placeholder="e.g. What is the formula for scaled dot-product attention? How does BERT differ from GPT?"]').first
        rag_inp.fill("Explain the difference between GPT and BERT based on the uploaded notes.")
        page.locator('button:has-text("Retrieve & Synthesize")').first.click()
        wait_for_idle(page)
        page.wait_for_selector("text=Grounded AI Response", timeout=20000)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 450)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig10_grounded_response.png"))

        # Figure 11: Diagram Explainer Vision Analysis
        print("Capturing Figure 11 (Diagram Explainer)...")
        click_nav(page, "Diagram Explainer")
        page.locator('button:has-text("Analyze Diagram")').first.click()
        wait_for_idle(page)
        page.wait_for_selector("text=Architectural Analysis Results", timeout=20000)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 400)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig11_diagram_explainer.png"))

        # Figure 12: Concept Image Studio & Illustration
        print("Capturing Figure 12 (Image Studio)...")
        click_nav(page, "Concept Image Studio")
        page.locator('button:has-text("Generate Educational Illustration")').first.click()
        wait_for_idle(page, timeout=35000)
        page.wait_for_selector("text=Generated Educational Diagram", timeout=20000)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 350)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig12_image_studio.png"))

        # Figure 13: PyTorch GAN Minimax Distribution & Loss
        print("Capturing Figure 13 (GAN Explorer)...")
        click_nav(page, "GAN Explorer")
        page.locator('button:has-text("Train GAN")').first.click()
        wait_for_idle(page)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 350)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig13_gan_explorer.png"))

        # Figure 14: Responsible AI Confusion Matrix & Grounding Verification
        print("Capturing Figure 14 (Responsible AI Confusion Matrix)...")
        click_nav(page, "Responsible AI")
        page.wait_for_selector(".js-plotly-plot", timeout=15000)
        page.wait_for_timeout(2000)
        page.mouse.move(750, 400)
        page.mouse.wheel(0, 250)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SCREENSHOTS_DIR / "fig14_responsible_ai.png"))

        browser.close()
        print("All 13 captioned screenshots successfully captured in:", SCREENSHOTS_DIR)


if __name__ == "__main__":
    capture_all()
