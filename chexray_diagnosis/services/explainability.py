"""
AI Explainability — Gemini-powered clinical explanations.

Generates patient-friendly explanations using Google Gemini 3.1 Flash Lite.
Falls back to comprehensive rule-based explanations when the API is
unavailable or no key is provided.

API keys are NEVER embedded — they're resolved through
``chexray_diagnosis.config`` at runtime.
"""

from typing import Dict, List, Optional

from chexray_diagnosis.config import get_gemini_api_key

# Lazy-import google-genai (optional dependency for fallback-only users)
_GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    pass


class ExplainabilityAI:
    """Human-friendly AI explanations with symptoms, precautions, and remedies."""

    MODEL_NAME = "gemini-3.1-flash-lite-preview"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = get_gemini_api_key(api_key)
        self.client = None

        if not _GENAI_AVAILABLE:
            print("⚠️  google-genai not installed — using rule-based fallback")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("✅ Gemini 3.1 Flash Lite initialized")
            except Exception as e:
                print(f"⚠️  Gemini init failed: {e}")
        else:
            print("⚠️  GEMINI_API_KEY not set — using rule-based fallback")

    # ------------------------------------------------------------------
    # NORMAL case
    # ------------------------------------------------------------------

    def generate_normal_explanation(self, model3_result: Dict) -> Dict:
        """Generate warm explanation for NORMAL results."""
        confidence = model3_result.get("confidence", 0)

        prompt = f"""You are a caring doctor explaining NORMAL chest X-ray results to a patient.

**PATIENT'S SITUATION:**
- Chest X-ray Result: **NORMAL** (no abnormalities detected)
- AI Confidence: {confidence:.1%}
- All lung, heart, and chest checks passed

**YOUR COMMUNICATION TASK:**
Write a warm, reassuring explanation that:
1. Celebrates the good news
2. Explains what "normal" means briefly
3. SYMPTOM AWARENESS — respiratory symptoms to watch for in the future
4. BASIC HOME REMEDIES for common respiratory issues
5. PREVENTIVE PRECAUTIONS — daily habits for lung health

**FORMATTING:** Use **bold** for medical terms. Bullet points for lists.
**DO NOT** say "Do you have any questions?" or invite follow-up.
**TONE:** Warm, educational, empowering.  **LENGTH:** 350-400 words.

Generate your explanation now:"""

        explanation = self._call_gemini(prompt, max_tokens=900, temperature=0.75)

        if not explanation:
            explanation = self._normal_fallback(confidence)
            api_used = "Rule-Based Fallback (Normal)"
        else:
            api_used = "Gemini 3.1 Flash Lite"

        return {
            "explanation": explanation,
            "api_used": api_used,
            "assessment": "NORMAL",
            "model3_input": model3_result,
        }

    # ------------------------------------------------------------------
    # ABNORMAL case
    # ------------------------------------------------------------------

    def generate_abnormal_explanation(
        self, model3_result: Dict, main_model_result: Dict
    ) -> Dict:
        """Generate empathetic explanation for ABNORMAL results."""
        confidence = model3_result.get("confidence", 0)
        pathologies = main_model_result.get("pathologies", [])
        assessment = main_model_result.get("assessment_level", "UNKNOWN")

        significant = sorted(
            [p for p in pathologies if p.get("probability", 0) > 0.3],
            key=lambda x: x.get("probability", 0),
            reverse=True,
        )

        findings_lines = "\n".join(
            f"- **{f.get('name', 'Unknown')}**: {f.get('probability', 0):.1%} likelihood"
            for f in significant[:6]
        ) or "- Abnormal patterns requiring clinical review"

        prompt = f"""You are a compassionate doctor explaining ABNORMAL chest X-ray results.

**PATIENT'S RESULTS:**
- Overall Status: **ABNORMAL** (AI confidence: {confidence:.1%})
- Severity: **{assessment}**
- Findings:
{findings_lines}

**TASK:** Write a clear, caring explanation:
1. Deliver the news honestly but kindly
2. Explain top 2-3 findings in plain language (use **bold** for condition names)
3. SPECIFIC SYMPTOMS to watch for based on THESE findings
4. PRECAUTIONS to take now
5. CLEAR NEXT STEPS — specialist, timeframe, expected tests

**DO NOT** say "Do you have any questions?" or invite follow-up.
**TONE:** Caring, direct, empowering.  **LENGTH:** 400-500 words.

Generate your patient-specific explanation now:"""

        explanation = self._call_gemini(prompt, max_tokens=1200, temperature=0.7)

        if not explanation:
            explanation = self._abnormal_fallback(confidence, assessment, significant)
            api_used = "Rule-Based Fallback (Abnormal)"
        else:
            api_used = "Gemini 3.1 Flash Lite"

        return {
            "explanation": explanation,
            "api_used": api_used,
            "assessment": "ABNORMAL",
            "model3_input": model3_result,
            "main_model_input": main_model_result,
        }

    # Legacy compat
    def generate_explanation(self, model3_result: Dict, main_model_result: Dict) -> Dict:
        prediction = model3_result.get("prediction", "").lower()
        if prediction == "normal":
            return self.generate_normal_explanation(model3_result)
        return self.generate_abnormal_explanation(model3_result, main_model_result)

    # ------------------------------------------------------------------
    # Gemini call
    # ------------------------------------------------------------------

    def _call_gemini(
        self, prompt: str, max_tokens: int = 900, temperature: float = 0.75
    ) -> Optional[str]:
        if not self.client:
            return None
        try:
            resp = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                ),
            )
            text = resp.text
            return text.strip() if text else None
        except Exception as e:
            print(f"⚠️  Gemini API call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_fallback(confidence: float) -> str:
        return f"""## 🎉 Great News — Your Chest X-ray is Normal!

Your chest X-ray results show everything is healthy. The AI analysis examined your X-ray with **{confidence:.1%} confidence** and found no signs of **pneumonia**, **fluid buildup**, **masses**, **heart enlargement**, or other conditions.

## 🔍 What We Checked
- **Lung tissue** — Clear, no infections
- **Heart size** — Normal
- **Pleural space** — No fluid
- **Airways** — Clear and open

## ⚠️ Symptoms to Watch For
**See a doctor within 24-48 hours if you develop:**
- **Persistent cough** lasting more than 2-3 weeks
- **Shortness of breath** during normal activities
- **Chest pain** that worsens with breathing
- **Coughing up blood**
- **Fever above 101°F** lasting more than 3 days

## 🏠 Basic Home Remedies
- **Steam inhalation** — loosens mucus naturally
- **Honey** (1-2 tsp) — soothes throat irritation
- **Hydration** — 8-10 glasses daily
- **Saltwater gargle** — reduces throat inflammation

## 💪 Keep Your Lungs Healthy
- **Never smoke or vape**
- **Exercise regularly** — 30 min daily
- **Monitor air quality** — stay indoors when AQI > 100
- **Stay vaccinated** — flu shot, COVID boosters, pneumonia vaccine
- **Eat lung-supporting foods** — berries, leafy greens, fatty fish

Your normal results are reassuring. Keep up with annual check-ups!"""

    @staticmethod
    def _abnormal_fallback(
        confidence: float, assessment: str, findings: List[Dict]
    ) -> str:
        findings_text = "\n".join(
            f"- **{f.get('name', 'Unknown')}** ({f.get('probability', 0):.1%})"
            for f in findings[:3]
        ) or "- Abnormal patterns requiring specialist review"

        symptom_map = {
            "Pneumonia": "**fever above 101°F**, **productive cough**, **chest pain** when breathing",
            "Pneumothorax": "**sudden sharp chest pain**, **severe shortness of breath**",
            "Effusion": "**difficulty breathing lying down**, **persistent dry cough**",
            "Cardiomegaly": "**swelling in legs/ankles**, **fatigue**, **shortness of breath**",
            "Atelectasis": "**shallow breathing**, **rapid breathing**, **low oxygen**",
        }

        symptoms = [
            symptom_map[f.get("name", "")]
            for f in findings[:3]
            if f.get("name", "") in symptom_map
        ]
        symptoms_text = (
            "Based on findings, watch for:\n\n" + "\n".join(f"- {s}" for s in symptoms)
            if symptoms
            else "Watch for:\n\n- **Worsening shortness of breath**\n- **Persistent chest pain**\n- **High fever**"
        )

        return f"""## 🩺 Important Findings in Your Chest X-ray

Your X-ray shows findings needing attention. AI confidence: **{confidence:.1%}**, level: **{assessment}**.

## 🔍 Detected
{findings_text}

## ⚠️ Symptoms to Watch — IMPORTANT
{symptoms_text}

**🚨 SEEK EMERGENCY CARE if you experience:**
- **Severe difficulty breathing**
- **Crushing chest pain** radiating to arm/jaw
- **Coughing up significant blood**
- **Blue lips or fingernails**

## 🛡️ Precautions Now
1. **Rest** — avoid strenuous activity
2. **Avoid irritants** — smoke, chemical fumes
3. **Sleep with head elevated**
4. **Stay hydrated** — 8-10 glasses daily
5. **Monitor temperature** twice daily

## 🏥 Next Steps
1. **Schedule specialist appointment** within 24-48 hours (pulmonologist or cardiologist)
2. **Gather medical records** — bring X-ray images
3. **Expect:** CT scan, blood tests, pulmonary function tests

Early detection + prompt care = best outcomes. You're on the right path."""
