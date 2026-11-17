import os
from config import Config

GROQ_AVAILABLE = False
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def get_breath_analysis_explanation(result, confidence, voc_percentage, patient_name="Patient", smoking_status="Unknown", symptoms="None", risk_level="Moderate"):
    """
    Get an AI-generated explanation for the diagnostic result using Groq LLM
    or clinical knowledge base fallback.
    """
    if GROQ_AVAILABLE and Config.GROQ_API_KEY:
        try:
            client = Groq(api_key=Config.GROQ_API_KEY)
            
            prompt = f"""
            You are a senior clinical oncologist AI integrated into a non-invasive lung cancer screening platform based on exhaled breath Volatile Organic Compound (VOC) biomarker profiling and pulmonary imaging.

Clinical Profile:
- Patient Reference: {patient_name}
- Prediction Category: {result}
- Model Confidence: {confidence}%
- Breath VOC Biomarker Level: {voc_percentage}%
- Smoking History: {smoking_status}
- Reported Symptoms: {symptoms}
- Calculated Clinical Risk: {risk_level}

Provide a concise, professional diagnostic summary (approx 180-220 words) covering:
1. Pathophysiological interpretation of {result} classification and {voc_percentage}% VOC level.
2. Immediate clinical next steps (e.g. low-dose CT, bronchoscopy, or routine pulmonary screening).
3. Risk mitigation & therapeutic pathways (e.g. targeted resection, radiation, or smoking cessation monitoring).
4. Clear clinical guidance written in an objective medical tone.
Do not use conversational filler or standard legal disclaimers.
            """
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            
            content = chat_completion.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content.strip()
        except Exception as e:
            print(f"Error calling Groq API: {e}")

    # Comprehensive Clinical Evidence Fallback
    if result == "Malignant":
        return (
            f"The AI biomarker evaluation indicates a Malignant profile with {confidence}% diagnostic certainty. "
            f"Exhaled VOC levels are elevated at {voc_percentage}%, reflecting abnormal cellular peroxidation and volatile alkane release typical of pulmonary neoplastic activity. "
            f"Given the patient's risk profile ({risk_level}) and smoking status ({smoking_status}), urgent contrast-enhanced High-Resolution CT (HRCT) of the chest and tissue biopsy (core needle or bronchoscopic EBUS) are warranted. "
            f"Multidisciplinary tumor board assessment is recommended to evaluate staging and determine suitability for surgical resection, stereotactic body radiation therapy (SBRT), or systemic targeted/immunotherapy."
        )
    elif result == "Benign":
        return (
            f"The screening assessment indicates a Benign pulmonary pattern with {confidence}% confidence. "
            f"The breath VOC level of {voc_percentage}% represents mild inflammatory or metabolic variability without malignant signature markers. "
            f"This often correlates with post-infectious granulomas, hamartomas, or chronic airway inflammation. "
            f"Recommended clinical follow-up includes a follow-up low-dose CT scan at 6 to 12 months to confirm structural stability, alongside pulmonary function testing and active risk-reduction protocols."
        )
    else: # Normal
        return (
            f"The diagnostic analysis reveals a Normal pulmonary biomarker spectrum with {confidence}% confidence. "
            f"Breath VOC concentrations ({voc_percentage}%) reside within baseline physiological thresholds. "
            f"No imaging or chemical evidence of nodular dysplasia or neoplastic activity is present. "
            f"Routine annual health screenings, continued cessation of environmental/tobacco smoke exposure, and active aerobic pulmonary maintenance are recommended."
        )
