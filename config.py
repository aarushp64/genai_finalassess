"""
NeuroTutor AI - Global Configuration
Author: Aarush Parate (Roll No: 1)
Project: P1 - AI-ML-DL Concept Tutor
Course: Generative AI Continuous Assessment
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_NOTES_DIR = DATA_DIR / "sample_notes"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_NOTES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# Application Metadata
APP_TITLE = "NeuroTutor AI"
APP_SUBTITLE = "AI–ML–DL Concept Tutor & Generative Intelligence Lab"
STUDENT_NAME = "Aarush Parate"
STUDENT_ROLL = "1"
PROJECT_CODE = "P1"
COURSE_NAME = "Generative AI"

# Theme Colors (Modern AI SaaS: Linear/Notion inspired)
THEME_COLORS = {
    "primary_bg": "#171D29",
    "sidebar": "#202838",
    "card_bg": "#293548",
    "secondary_surface": "#303E52",
    "accent": "#8BAFD0",
    "accent_hover": "#A4C4E2",
    "main_text": "#F2F4F7",
    "secondary_text": "#AAB5C4",
    "border": "#39465A",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "card_border": "#39465A",
}

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM Defaults
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.7
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.5
MAX_HISTORY_TURNS = 10

# RAG & Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K_RETRIEVAL = 3

# Attention Visualizer Model
ATTENTION_MODEL_NAME = "distilbert-base-uncased"

# Learning Levels
LEARNING_LEVELS = {
    "School Level": {
        "title": "School Level (Intuitive & Everyday Analogies)",
        "description": "Simple explanations, relatable everyday analogies (kitchen recipes, sorting legos), beginner-friendly examples, and zero intimidating jargon.",
        "badge_color": "#4CAF50"
    },
    "Undergraduate Level": {
        "title": "Undergraduate Level (Algorithmic & Mathematical Intuition)",
        "description": "Technical explanations, algorithmic pseudocode, mathematical intuition, network architecture diagrams, and practical coding examples.",
        "badge_color": "#2196F3"
    },
    "Expert Level": {
        "title": "Expert Level (Research, Math Formulations & Scaling)",
        "description": "Exhaustive technical formulations, rigorous mathematical derivations, loss functions, attention complexity, production scaling trade-offs, and research insights.",
        "badge_color": "#9C27B0"
    }
}
