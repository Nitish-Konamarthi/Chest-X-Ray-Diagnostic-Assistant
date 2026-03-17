"""
Explainability AI Module - FINAL OPTIMIZED VERSION
===================================================

Features:
- Human-friendly, conversational tone
- Important medical terms highlighted in bold
- Symptom awareness and monitoring guidance
- Practical precautions and remedies
- NO "Do you have questions?" - complete, self-contained explanations

NORMAL cases: Symptom awareness + basic remedies + preventive care
ABNORMAL cases: Specific symptoms to watch + precautions + clear next steps
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_dotenv_path = find_dotenv(usecwd=False)
if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _env_file = os.path.join(_project_root, '.env')
    if os.path.exists(_env_file):
        load_dotenv(_env_file)

# Import google-genai
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print("⚠️  google-genai not installed — run: pip install google-genai")


class ExplainabilityAI:
    """Human-friendly AI explanations with symptoms, precautions, and remedies"""

    MODEL_NAME = "gemini-3.1-flash-lite-preview"

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = None

        if not _GENAI_AVAILABLE:
            print("⚠️  google-genai not available — AI explanations will use rule-based fallback")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"✅ Gemini 3.1 Flash Lite initialized successfully")
            except Exception as e:
                print(f"⚠️  Gemini API initialization failed: {e}")
                self.client = None
        else:
            print("⚠️  GEMINI_API_KEY not found — AI explanations will use rule-based fallback")

    # ============================================================================
    # NORMAL CASE - Symptom Awareness + Basic Remedies + Prevention
    # ============================================================================
    
    def generate_normal_explanation(self, model3_result: Dict) -> Dict:
        """
        Generate warm explanation for NORMAL results with symptom awareness.
        Focus: "Even though you're healthy now, here's what to watch for"
        """
        model3_confidence = model3_result.get('confidence', 0)
        
        prompt = f"""You are a caring doctor explaining NORMAL chest X-ray results to a patient.

**PATIENT'S SITUATION:**
- Chest X-ray Result: **NORMAL** (no abnormalities detected)
- AI Confidence: {model3_confidence:.1%}
- All lung, heart, and chest checks passed

**YOUR COMMUNICATION TASK:**

Write a warm, reassuring explanation that:

1. **Celebrates the good news** - Start positive! Their lungs and heart look healthy.

2. **Explains what "normal" means** - What did we check? What was ruled out? Keep it brief and reassuring.

3. **SYMPTOM AWARENESS** (Critical section) - Even though they're healthy NOW, educate them about respiratory symptoms to watch for in the FUTURE:
   - What symptoms could indicate lung problems developing
   - What symptoms need immediate attention vs routine check-up
   - How to recognize early warning signs
   - Frame this as "staying informed" not "being scared"

4. **BASIC HOME REMEDIES** for common respiratory issues (cough, congestion, mild chest discomfort):
   - Natural remedies that are safe and evidence-based
   - When home remedies are appropriate vs when to see a doctor
   - Practical, easy-to-do suggestions

5. **PREVENTIVE PRECAUTIONS** - Daily habits to keep lungs healthy:
   - Most impactful actions (smoking, exercise, air quality)
   - Environmental precautions
   - Nutrition tips
   - Vaccination reminders

**FORMATTING REQUIREMENTS:**
- Use **bold text** for important medical terms (e.g., **pneumonia**, **shortness of breath**)
- Use **bold** for any symptom names or condition names
- Use bullet points for lists (symptoms, remedies, precautions)
- Keep sections clear but conversational

**CRITICAL INSTRUCTIONS:**
- DO NOT say "Do you have any questions?"
- DO NOT say "Feel free to ask me..."
- DO NOT invite follow-up questions
- END with encouragement about staying healthy
- Make it COMPLETE - they shouldn't need to ask questions

**TONE:** Warm, friendly, educational, empowering
**LENGTH:** 350-400 words
**STYLE:** Like a caring doctor giving comprehensive health guidance

Generate your explanation now:"""

        explanation = None
        api_used = None

        # Try Gemini API
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=900,
                        temperature=0.75,
                        top_p=0.95,
                    )
                )
                text = response.text
                if text:
                    explanation = text.strip()
                    api_used = "Gemini 3.1 Flash Lite"
                    print(f"✅ Generated normal case with symptom awareness ({len(text)} chars)")
            except Exception as e:
                print(f"⚠️  Gemini API call failed: {e}")

        # Fallback
        if not explanation:
            explanation = f"""## 🎉 Great News - Your Chest X-ray is Normal!

Your chest X-ray results are in, and everything looks healthy! The AI analysis examined your X-ray with **{model3_confidence:.1%} confidence** and found no signs of **pneumonia**, **fluid buildup**, **masses**, **heart enlargement**, or other common chest conditions. Your lungs, heart, and chest structure are all within normal, healthy limits.

## 🔍 What We Checked

- **Lung tissue** - Clear, no infections or abnormal spots
- **Heart size and position** - Normal and healthy
- **Pleural space** - No fluid accumulation
- **Chest bones** - No fractures or issues
- **Airways** - Clear and open

## ⚠️ Symptoms to Watch For (Stay Informed)

Even though you're healthy now, it's good to know when respiratory symptoms need attention:

**See a doctor within 24-48 hours if you develop:**
- **Persistent cough** lasting more than 2-3 weeks
- **Shortness of breath** during normal activities or at rest
- **Chest pain** that worsens with breathing or doesn't go away
- **Coughing up blood** (even small amounts)
- **Fever above 101°F (38.3°C)** lasting more than 3 days
- **Wheezing** or difficulty breathing that's new or worsening

**Routine check-up needed for:**
- Persistent **congestion** or **post-nasal drip** beyond 2 weeks
- Mild **chest tightness** that comes and goes
- Changes in your normal **breathing patterns**

## 🏠 Basic Remedies for Common Respiratory Issues

If you develop a mild cough, congestion, or chest discomfort:

**Natural remedies that help:**
- **Steam inhalation** - Breathe warm, moist air (shower steam, bowl of hot water). Loosens mucus naturally.
- **Honey** (1-2 teaspoons) - Soothes throat irritation and reduces coughing. Don't give to children under 1 year.
- **Hydration** - Drink 8-10 glasses of water daily. Thins mucus and keeps airways moist.
- **Saltwater gargle** - Reduces throat inflammation and kills bacteria.
- **Elevate head while sleeping** - Helps drainage and reduces nighttime coughing.
- **Humidifier** - Adds moisture to air, especially helpful in winter.

**When NOT to rely on home remedies:**
- If symptoms worsen after 3-4 days
- If you have difficulty breathing
- If you develop high fever or severe chest pain
- If you have underlying health conditions (asthma, heart disease, diabetes)

## 💪 Daily Precautions to Keep Your Lungs Healthy

**Top preventive actions:**

- **Never smoke or vape** - Single most important thing for lung health. If you smoke, talk to your doctor about quitting programs.
  
- **Exercise regularly** - 30 minutes daily of brisk walking, swimming, or cycling strengthens respiratory muscles and improves lung capacity.

- **Monitor air quality** - Check AQI (Air Quality Index) before outdoor activities. Stay indoors when AQI is above 100. Use HEPA air purifiers at home.

- **Practice good hygiene** - Wash hands frequently (20 seconds with soap). Avoid touching your face. Stay away from sick people when possible.

- **Stay vaccinated** - Get annual **flu shot**, **COVID-19 boosters**, and **pneumonia vaccine** (if 65+ or high-risk). Prevents serious respiratory infections.

- **Eat lung-supporting foods** - Berries (antioxidants), leafy greens (vitamins), fatty fish (omega-3s), nuts (vitamin E), citrus fruits (vitamin C). These support respiratory health.

- **Avoid respiratory irritants** - Cleaning chemical fumes, paint fumes, strong perfumes. Use masks when exposure is unavoidable.

## 📅 Moving Forward

Your normal X-ray results are reassuring. You likely won't need another chest X-ray unless symptoms develop or your doctor recommends one for another reason. Keep up with annual health check-ups with your primary care physician.

Remember, maintaining these healthy habits now means staying ahead of potential problems. Your lungs work hard every day - take care of them, and they'll take care of you!"""
            api_used = "Rule-Based Fallback (Normal)"

        return {
            "explanation": explanation,
            "api_used": api_used,
            "assessment": "NORMAL",
            "model3_input": model3_result,
        }

    # ============================================================================
    # ABNORMAL CASE - Specific Symptoms + Precautions + Clear Action Steps
    # ============================================================================
    
    def generate_abnormal_explanation(self, model3_result: Dict, main_model_result: Dict) -> Dict:
        """
        Generate empathetic explanation for ABNORMAL results.
        Focus: Specific symptoms for THESE findings + precautions + what to do
        """
        model3_confidence = model3_result.get('confidence', 0)
        pathologies = main_model_result.get('pathologies', [])
        assessment = main_model_result.get('assessment_level', 'UNKNOWN')
        
        # Get significant findings
        significant = [p for p in pathologies if p.get('probability', 0) > 0.3]
        significant.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        findings_data = ""
        for finding in significant[:6]:
            name = finding.get('name', 'Unknown')
            prob = finding.get('probability', 0)
            findings_data += f"- **{name}**: {prob:.1%} likelihood\n"
        
        if not findings_data:
            findings_data = "- Abnormal patterns requiring clinical review\n"

        prompt = f"""You are a compassionate doctor explaining ABNORMAL chest X-ray results to a patient.

**PATIENT'S RESULTS:**
- Overall Status: **ABNORMAL** (AI confidence: {model3_confidence:.1%})
- Severity Assessment: **{assessment}**
- Detected Findings:
{findings_data}

**YOUR COMMUNICATION TASK:**

Write a clear, caring explanation that:

1. **Deliver the news honestly but kindly** - State that the X-ray shows findings needing medical attention. Be direct but not scary. Acknowledge this might be worrying but emphasize early detection is good.

2. **Explain the findings in plain language** - For the TOP 2-3 findings detected:
   - What is this condition? (simple explanation)
   - What does it mean? (what's happening in their body)
   - How serious is it? (context for the likelihood %)
   - Use **bold** for all medical condition names

3. **SPECIFIC SYMPTOMS TO WATCH FOR** (Critical section) - Based on THESE specific findings, what symptoms indicate:
   - The condition is worsening → seek immediate care
   - The condition is stable → mention at next appointment
   - Emergency warning signs → go to ER immediately
   - Be specific to the detected conditions, not generic

4. **PRECAUTIONS TO TAKE NOW** - Practical actions before seeing the specialist:
   - Activity modifications (rest, avoid exertion, etc.)
   - Environmental precautions (avoid smoke, pollutants)
   - Dietary suggestions if relevant
   - What NOT to do (don't ignore symptoms, don't delay appointment)

5. **CLEAR NEXT STEPS** - Prioritized action plan:
   - What specialist to see (pulmonologist, cardiologist, etc.)
   - Timeframe urgency (24-48 hours, within 1 week, etc.)
   - What tests to expect (CT scan, blood work, etc.)
   - How to prepare for the appointment

**FORMATTING REQUIREMENTS:**
- Use **bold** for ALL medical terms (condition names, symptoms, test names)
- Use **bold** for urgency indicators (URGENT, IMMEDIATE, etc.)
- Use bullet points for symptom lists and action steps
- Make important warnings stand out

**CRITICAL INSTRUCTIONS:**
- DO NOT say "Do you have any questions?"
- DO NOT say "Let me know if..."
- DO NOT invite follow-up
- END with reassurance about early detection and treatability
- Be complete and thorough - cover everything they need to know

**TONE:** Caring, direct, empowering, balanced (honest but hopeful)
**LENGTH:** 400-500 words
**STYLE:** Like a doctor having an important but caring conversation

Generate your patient-specific explanation now:"""

        explanation = None
        api_used = None

        # Try Gemini API
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=1200,
                        temperature=0.7,
                        top_p=0.95,
                    )
                )
                text = response.text
                if text:
                    explanation = text.strip()
                    api_used = "Gemini 3.1 Flash Lite"
                    print(f"✅ Generated abnormal case with specific symptoms & precautions ({len(text)} chars)")
            except Exception as e:
                print(f"⚠️  Gemini API call failed: {e}")

        # Fallback
        if not explanation:
            explanation = self._create_abnormal_fallback(model3_confidence, assessment, significant)
            api_used = "Rule-Based Fallback (Abnormal)"

        return {
            "explanation": explanation,
            "api_used": api_used,
            "assessment": "ABNORMAL",
            "model3_input": model3_result,
            "main_model_input": main_model_result,
        }

    def _create_abnormal_fallback(self, confidence: float, assessment: str, findings: List[Dict]) -> str:
        """Fallback with symptoms and precautions"""
        
        findings_text = ""
        for finding in findings[:3]:
            name = finding.get('name', 'Unknown')
            prob = finding.get('probability', 0)
            findings_text += f"- **{name}** ({prob:.1%} likelihood)\n"
        
        if not findings_text:
            findings_text = "- Abnormal patterns requiring specialist review\n"

        # Map findings to specific symptoms
        symptom_guidance = {
            'Pneumonia': '**fever above 101°F**, **productive cough**, **chest pain** when breathing',
            'Pneumothorax': '**sudden sharp chest pain**, **severe shortness of breath**, **rapid heart rate**',
            'Effusion': '**difficulty breathing when lying down**, **persistent dry cough**, **chest heaviness**',
            'Cardiomegaly': '**swelling in legs/ankles**, **fatigue**, **shortness of breath** with mild activity',
            'Atelectasis': '**shallow breathing**, **rapid breathing**, **low oxygen levels**',
        }
        
        # Get relevant symptoms based on findings
        relevant_symptoms = []
        for finding in findings[:3]:
            name = finding.get('name', '')
            if name in symptom_guidance:
                relevant_symptoms.append(symptom_guidance[name])

        symptoms_section = ""
        if relevant_symptoms:
            symptoms_section = "Based on the detected findings, watch carefully for:\n\n" + "\n".join([f"- {s}" for s in relevant_symptoms])
        else:
            symptoms_section = "Watch for:\n\n- **Worsening shortness of breath**\n- **Persistent chest pain**\n- **High fever** (above 101°F/38.3°C)\n- **Coughing up blood**\n- **Rapid heart rate or palpitations**"

        return f"""## 🩺 Important Findings in Your Chest X-ray

Your chest X-ray shows findings that need medical attention. The AI analysis identified abnormal patterns with **{confidence:.1%} confidence**, classified as **{assessment}** level.

I want to be direct with you: seeing "abnormal" on a medical report can feel scary. But here's what's important - we've found these issues early, which significantly improves treatment outcomes. Let me explain what we found and what you need to do.

## 🔍 What We Detected

{findings_text.strip()}

These findings tell us that something in your chest needs closer evaluation by a medical specialist. The percentages indicate how confident the AI is about each finding, but only a specialist can confirm the diagnosis with additional tests.

## ⚠️ Symptoms to Watch For - **IMPORTANT**

{symptoms_section}

**🚨 SEEK IMMEDIATE EMERGENCY CARE (Call 911 or go to ER) if you experience:**
- **Severe difficulty breathing** or inability to catch your breath
- **Chest pain** that feels crushing or radiates to arm/jaw
- **Coughing up significant blood**
- **Confusion** or **loss of consciousness**
- **Blue lips or fingernails** (sign of low oxygen)

**📞 CALL YOUR DOCTOR IMMEDIATELY (Same Day) if you notice:**
- **Worsening symptoms** from when the X-ray was taken
- **New fever** above 101°F (38.3°C)
- **Increased difficulty breathing** even at rest
- **Significant chest discomfort** that's getting worse

## 🛡️ Precautions to Take Now (Before Your Specialist Appointment)

**Do these immediately:**

1. **Rest and avoid overexertion** - Don't do strenuous exercise or heavy lifting until cleared by your doctor. Your body needs energy to heal.

2. **Avoid respiratory irritants** - Stay away from smoke (including secondhand smoke), strong chemical fumes, dust, and air pollution. These can worsen your condition.

3. **Sleep with head elevated** - Use 2-3 pillows or raise the head of your bed. This helps with breathing and reduces strain.

4. **Stay hydrated** - Drink 8-10 glasses of water daily unless your doctor restricts fluids. Helps thin mucus and supports healing.

5. **Monitor your temperature** - Check it twice daily. Write down any readings above 100.4°F.

6. **Don't ignore symptoms** - If anything worsens, don't "wait and see" - contact your doctor immediately.

**What NOT to do:**
- ❌ Don't smoke or vape
- ❌ Don't delay making your specialist appointment
- ❌ Don't take any new medications without doctor approval
- ❌ Don't fly or travel to high altitudes until cleared

## 🏥 Your Action Plan - Next Steps

**URGENT - Do within 24-48 hours:**

1. **Schedule specialist appointment** - You need to see a **pulmonologist** (lung specialist) or **cardiologist** (heart specialist) depending on the findings. Call your primary care doctor for a referral, or if they're unavailable, call the specialist directly and mention you have abnormal X-ray results.

2. **Gather your medical records** - Bring your X-ray images and any previous chest X-rays if you have them. This helps the specialist compare.

**Expect these additional tests:**
- **CT scan of chest** - More detailed imaging to see exactly what's happening
- **Blood tests** - To check for infection, inflammation, or other issues
- **Pulmonary function tests** - Measures how well your lungs work
- **Possibly an ECG or echocardiogram** - If heart-related findings

**Timeline:** Aim to see a specialist within **1-2 weeks maximum**. If you have worsening symptoms, push for an earlier appointment or go to urgent care.

## 💚 Important Perspective

Many chest X-ray findings, especially when caught early like yours, are very manageable with proper treatment. **Pneumonia** responds well to antibiotics. **Pleural effusion** can often be drained. **Atelectasis** often improves with breathing exercises and treatment.

The key is: you've taken the right step by getting this X-ray. Now take the next step by seeing a specialist promptly. Modern medicine has excellent diagnostic tools and treatments for these conditions.

This AI analysis has identified areas needing attention. Your specialist will provide the definitive diagnosis, treatment plan, and answer all your specific questions during your appointment.

**Remember:** Early detection + prompt specialist care = best possible outcomes. You're on the right path."""

    # Legacy method for backward compatibility
    def generate_explanation(self, model3_result: Dict, main_model_result: Dict) -> Dict:
        """Backward compatibility"""
        prediction = model3_result.get('prediction', '').lower()
        
        if prediction == 'normal':
            return self.generate_normal_explanation(model3_result)
        else:
            return self.generate_abnormal_explanation(model3_result, main_model_result)


# Global instance
explainability_ai = ExplainabilityAI()