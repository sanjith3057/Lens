import fitz  # PyMuPDF
import re
import os
from PIL import Image
import io
from src.__init__ import logger

def sanitize_input(user_input):
    # Prompt Injection Prevention
    forbidden = [
        'ignore previous instructions', 'system prompt', 'act as', 'override',
        'you are now', 'disregard', 'forget everything', 'new rule'
    ]
    
    input_lower = user_input.lower()
    if any(f in input_lower for f in forbidden):
        logger.warning(f"Potential prompt injection detected: '{user_input}'")
        return f"[SECURE BLOCK] Input contained forbidden instructional patterns. Original query: {user_input[:50]}..."
    
    # Character-level manipulation check
    if '"""' in user_input or "'''" in user_input:
         logger.warning("Triple quotes detected - potential injection attempt.")
         user_input = user_input.replace('"""', '"').replace("'''", "'")

    return user_input

def redact_pii(text):
    patterns = [
        (r'\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b', '[REDACTED-SSN/PHONE]'),
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED-CREDIT]'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED-EMAIL]'),
        (r'\+\d{1,3}[\s-]?\d{10,12}\b', '[REDACTED-INTL-PHONE]')
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text

class MultimodalParser:
    """Layer 1: Multimodal Document Parser using PyMuPDF (fitz)"""
    
    @staticmethod
    def parse_pdf(file_path):
        if not file_path.endswith('.pdf'):
            return None, "Error: Invalid file format (PDF required)"

        pages_data = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract Text
                text = page.get_text()
                
                # Extract Images
                image_list = page.get_images(full=True)
                images = []
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
                
                # Fallback: Capture page screenshot for complex layouts if text is sparse
                if len(text.strip()) < 50 and len(image_list) == 0:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    images.append(img_data)

                pages_data.append({
                    "page": page_num + 1,
                    "text": text,
                    "images": images
                })
            
            doc.close()
            return pages_data, None
        except Exception as e:
            logger.error(f"Multimodal Parser Error: {e}")
            return None, f"Parsing error: {str(e)}"
