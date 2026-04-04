import base64
import requests
from io import BytesIO
from PIL import Image
from src.__init__ import logger, SecureConfig

def sanitize_prompt(prompt):
    dangerous = ['<system>', 'system', 'role=assistant']
    for d in dangerous:
        if d in prompt:
            logger.warning(f"Sanitizing prompt injection: '{d}'")
            prompt = f"User: {prompt}"
    return prompt

class VisionLanguageClient:
    """Layer 3: Lightweight Visual Reasoning (Llama 4 Scout 17B-16E MoE Upgrade)"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.api_key = api_key
        self.model = model

    def generate(self, text: str, image_bytes: bytes):
        # 1. Structure the query for Vision-capable MoE
        safe_query = sanitize_prompt(text)
        prompt = (
            f"Question: {safe_query}\n"
            "Analyze the visual details. If this is a document, read the text. "
            "Respond in 2 concise sentences."
        )

        # 2. Process Image (Force high stability: 600px JPEG)
        try:
            img = Image.open(BytesIO(image_bytes))
            img = img.convert("RGB")
            
            # 600px is the stability sweet spot for Groq MoE
            img.thumbnail((600, 600))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=65)
            payload_bytes = buffer.getvalue()
        except Exception as e:
            logger.error(f"VLM Pre-process Fail: {e}")
            return {"answer": f"VLM Fail: {e}", "status": "error"}

        # 3. Base64 (Clean)
        base64_image = base64.b64encode(payload_bytes).decode('utf-8').strip()

        # 4. Construct Payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0 # Force deterministic behavior
        }

        # 5. Send Request
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = { 'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json' }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code != 200:
                msg = response.text
                return {"answer": f"API Fail ({response.status_code}): {msg}", "status": "error"}
            
            result = response.json()
            # Success feedback in terminal
            print(f"--- Llama 4 VLM Success: {self.model} responded in {response.elapsed.total_seconds():.2f}s ---")
            return {"answer": result['choices'][0]['message']['content'], "status": "success"}
        except Exception as e:
            return {"answer": f"Connection Fail: {e}", "status": "error"}
