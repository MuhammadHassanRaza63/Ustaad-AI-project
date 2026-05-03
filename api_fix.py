# Simplified Google GenAI function using the current Gemini REST endpoint.

def get_concept_explanation_fixed(prompt, grade, subject, topic, language, api_key, selected_model):
    """Fixed API handler for Google Generative AI"""
    
    import requests
    
    lang_instructions = {
        "Roman Urdu": "Answer in Roman Urdu (Hinglish). Use simple, everyday examples from Pakistan.",
        "English": "Answer in clear English. Use simple examples suitable for a high school student.",
        "Urdu": "جواب اردو میں دیں۔ سادہ اردو میں سمجھائیں۔"
    }

    system_prompt = f"""You are an expert science tutor for {grade}th grade {subject} in Pakistan schools.

Your task:
- Explain concepts clearly using local Pakistani examples
- Break complex ideas into simple steps
- Use real-life analogies students can relate to
- Avoid rote memorization; focus on understanding
- For {topic}: provide step-by-step explanation

{lang_instructions.get(language, lang_instructions['Roman Urdu'])}

Keep answers concise (2-3 paragraphs max), engaging, and suitable for a high school student."""

    user_message = f"Question: {prompt}"
    
    last_error = "No response"
    preferred_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    models = [selected_model] if selected_model else []
    models.extend(model for model in preferred_models if model not in models)

    for model in [m for m in models if m]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"{system_prompt}\n\n{user_message}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1536,
                }
            }
            if model.startswith("gemini-2.5"):
                payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts).strip()
                if text:
                    return text
                last_error = "Empty response"
            else:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    detail = response.text
                last_error = f"HTTP {response.status_code}: {detail[:180]}"
        except Exception as e:
            last_error = str(e)[:180]

    raise Exception(f"API Error: {last_error}")
