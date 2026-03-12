"""
Explainability AI Module
========================

Provides AI-powered explanations for medical diagnoses using Gemini 2.0 Flash.
Uses the new Google Gen AI SDK (google-genai) — the modern replacement for
the deprecated google-generativeai library.

Features:
- Generates structured clinical explanations with clearly labelled sections
- Uses Gemini 2.0 Flash (latest Google AI Studio API as of 2025)
- Produces Markdown output for rich frontend rendering
- Falls back to a structured rule-based explanation if Gemini is unavailable
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Load environment variables — resolve from project root regardless of cwd
# ---------------------------------------------------------------------------
_dotenv_path = find_dotenv(usecwd=False)
if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _env_file = os.path.join(_project_root, '.env')
    if os.path.exists(_env_file):
        load_dotenv(_env_file)

# ---------------------------------------------------------------------------
# Import google-genai (new SDK) — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print("⚠️  google-genai not installed — run: pip install google-genai")


class ExplainabilityAI:
    """AI-powered explainability for medical diagnoses using Gemini 2.0 Flash"""

    MODEL_NAME = "gemini-2.0-flash"

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = None

        if not _GENAI_AVAILABLE:
            print("⚠️  google-genai not available — AI explanations will use rule-based fallback")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"✅ Gemini 2.0 Flash client initialized successfully")
            except Exception as e:
                print(f"⚠️  Gemini API initialization failed: {e}")
                self.client = None
        else:
            print("⚠️  GEMINI_API_KEY not found — AI explanations will use rule-based fallback")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _create_prompt(self, model3_result: Dict, main_model_result: Dict) -> str:
        """Build a structured Markdown prompt for Gemini"""

        model3_status = "NORMAL" if model3_result.get('prediction', '').lower() == 'normal' else "ABNORMAL"
        model3_confidence = model3_result.get('confidence', 0)

        pathologies = main_model_result.get('pathologies', [])
        significant = [p for p in pathologies if p.get('probability', 0) > 0.3]

        findings_text = "\n".join([
            f"- **{p['name']}**: {p['probability']:.1%} probability"
            for p in significant[:6]
        ]) if significant else "- No significant findings above 30% threshold"

        assessment = main_model_result.get('assessment_level', 'UNKNOWN')

        prompt = f"""You are an expert medical AI assistant specialising in chest radiology.

A chest X-ray has been analysed by a deep-learning pipeline. Your task is to produce a clear, structured clinical explanation for a **patient or a clinician**. Use the data below.

---
**Binary Classifier Result:** {model3_status} (confidence: {model3_confidence:.1%})
**CheXNet Assessment Level:** {assessment}

**Significant Pathology Findings:**
{findings_text}
---

Respond ONLY in the following Markdown structure — keep each section concise:

## 🩺 Overall Assessment
One sentence stating whether the X-ray appears NORMAL or ABNORMAL and the confidence level.

## 🔍 Key Findings
Bullet-point summary of the most significant detected conditions, their likelihood, and what they mean clinically.

## 📋 Recommendations
2-3 actionable next steps (e.g., follow-up imaging, specialist referral, observation).

## ⚠️ Disclaimer
One sentence reminding the reader this is an AI-generated screening aid and should be confirmed by a licensed radiologist.

Keep the total response under 300 words. Use plain language understandable by patients while remaining medically accurate."""

        return prompt.strip()

    # ------------------------------------------------------------------
    # Gemini API call
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini 2.0 Flash using the new google-genai client SDK"""
        if not self.client:
            return None

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=600,
                    temperature=0.3,
                )
            )
            text = response.text
            if text:
                return text.strip()
            return None
        except Exception as e:
            print(f"⚠️  Gemini API call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Rule-based fallback — same Markdown section structure as Gemini
    # ------------------------------------------------------------------
    def _create_default_explanation(self, model3_result: Dict, main_model_result: Dict) -> str:
        """Structured rule-based fallback when Gemini is unavailable"""

        model3_status = "NORMAL" if model3_result.get('prediction', '').lower() == 'normal' else "ABNORMAL"
        model3_confidence = model3_result.get('confidence', 0)
        assessment = main_model_result.get('assessment_level', 'UNKNOWN')
        pathologies = main_model_result.get('pathologies', [])
        significant = [p for p in pathologies if p.get('probability', 0) > 0.4]

        findings_lines = "\n".join([
            f"- **{p['name']}** — {p['probability']:.1%} probability"
            for p in significant[:5]
        ]) if significant else "- No findings above the 40% clinical threshold"

        normal_text = (
            "The X-ray appears within normal limits for all evaluated conditions. "
            "No significant pathological findings were detected by the automated system."
        )
        abnormal_text = (
            "The X-ray shows potential abnormalities that warrant clinical attention. "
            "Please review the detailed probability scores and consult a radiologist."
        )

        rec_normal = (
            "- Routine follow-up as clinically indicated\n"
            "- Report any new or worsening respiratory symptoms to your physician\n"
            "- Maintain regular health screening schedules"
        )
        rec_abnormal = (
            "- Urgent review by a qualified radiologist is recommended\n"
            "- Correlate with patient history, physical examination, and laboratory findings\n"
            "- Consider follow-up imaging or appropriate specialist referral"
        )

        is_normal = assessment == "NORMAL"

        return f"""## 🩺 Overall Assessment
The chest X-ray has been classified as **{model3_status}** (confidence: {model3_confidence:.1%}) by the binary classification model. The CheXNet pathology analysis indicates **{assessment}** findings.

## 🔍 Key Findings
{findings_lines}

{normal_text if is_normal else abnormal_text}

## 📋 Recommendations
{rec_normal if is_normal else rec_abnormal}

## ⚠️ Disclaimer
This analysis is generated by an AI screening system and is intended to **assist — not replace** — qualified radiologists and healthcare professionals. All findings must be clinically correlated and confirmed by a licensed physician."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_explanation(self, model3_result: Dict, main_model_result: Dict) -> Dict:
        """
        Generate AI-powered clinical explanation.

        Args:
            model3_result: Binary classification result (normal/abnormal)
            main_model_result: Main CheXNet model result with pathologies

        Returns:
            Dict with 'explanation' (Markdown string) and 'api_used' metadata
        """
        prompt = self._create_prompt(model3_result, main_model_result)

        explanation = None
        api_used = None

        if self.client:
            explanation = self._call_gemini(prompt)
            if explanation:
                api_used = "Gemini 2.0 Flash"

        if not explanation:
            explanation = self._create_default_explanation(model3_result, main_model_result)
            api_used = "Rule-Based Fallback"

        return {
            "explanation": explanation,
            "api_used": api_used,
            "model3_input": model3_result,
            "main_model_input": main_model_result,
        }


# Global instance — imported by app.py
explainability_ai = ExplainabilityAI()