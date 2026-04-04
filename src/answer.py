import os
import concurrent.futures
from typing import Optional, List
import json
import torch
import numpy as np
from src.parser import MultimodalParser, sanitize_input, redact_pii
from src.embedding import CLIPEmbedder
from src.vlq import VisionLanguageClient
from src.__init__ import logger, SecureConfig

def chunk_text(text: str, chunk_size: int = 150) -> list:
    """Helper to split text into CLIP-safe 150-char chunks for 100% stability"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

class AnswerGenerator:
    """Layer 4: Advanced Multimodal Semantic Synthesis (Llama 4 Parallel Upgrade)"""
    
    def __init__(self):
        # 1. Initialize Multimodal Layer 2 (Local CLIP on GPU)
        self.embedder = CLIPEmbedder()
        
        # 2. Initialize Layer 3 (Vision) - Using Llama 4 Scout
        self.vl_client = VisionLanguageClient(SecureConfig.get_api_key('GROQ_VISION_API_KEY'))
        
        # 3. Final Synthesis LLM (Groq) - Using Llama 4 Scout
        self.api_key = SecureConfig.get_api_key('GROQ_VISION_API_KEY')
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def process_image_node(self, img_res, query):
        """Worker function for concurrent vision reasoning"""
        source = "Manual Image" if img_res['type'] == "manual" else f"Page {img_res['page']}"
        try:
            vlm_res = self.vl_client.generate(query, img_res['bytes'])
            if vlm_res.get("status") == "success":
                return f"[VISUAL_SOURCE ({source})]: {vlm_res.get('answer')}"
            else:
                return f"[VISUAL_SOURCE ({source})]: Analysis skipped due to API availability."
        except Exception as e:
            return f"[VISUAL_SOURCE ({source})]: Fail: {str(e)}"

    def generate(self, file_path: Optional[str], query: str, manual_images: list = None) -> str:
        safe_query = sanitize_input(query)
        query_vec = self.embedder.embed_text(safe_query)
        
        page_chunks = {} # page_num -> list of (chunk_text, score)
        image_to_rank = []
        full_page_text_map = {} # page_num -> full text
        
        # 1. Layer 1 & 2: Process PDF content (optional)
        if file_path and os.path.exists(file_path):
            pages_data, error = MultimodalParser.parse_pdf(file_path)
            if not error:
                for p in pages_data[:15]: # Scan 15 pages
                    clean_text = redact_pii(p['text'])
                    full_page_text_map[p['page']] = clean_text # Store for context
                    
                    # Rank Chunks for Page Importance
                    chunks = chunk_text(clean_text)
                    page_scores = []
                    for c in chunks:
                        if len(c.strip()) > 5:
                            text_vec = self.embedder.embed_text(c)
                            score = self.embedder.similarity(query_vec, text_vec)
                            page_scores.append(score)
                    
                    if page_scores:
                        page_chunks[p['page']] = max(page_scores) # Peak relevance of this page
                    
                    # Store PDF images for ranking
                    for img_bytes in p['images']:
                        img_vec = self.embedder.embed_image(img_bytes)
                        score = self.embedder.similarity(query_vec, img_vec)
                        image_to_rank.append({"page": p['page'], "bytes": img_bytes, "score": score, "type": "pdf"})

        # 2. Layer 2: Process Manual Uploaded Images (Force high priority)
        if manual_images:
            for idx, img_bytes in enumerate(manual_images):
                img_vec = self.embedder.embed_image(img_bytes)
                score = self.embedder.similarity(query_vec, img_vec)
                # Boost manual images to Top Priority
                image_to_rank.append({"page": f"{idx+1}", "bytes": img_bytes, "score": score + 0.5, "type": "manual"})

        # 3. Layer 3: PARALLEL Visual Reasoning Logic
        # Identify top pages by score (Top 3)
        top_page_nums = sorted(page_chunks.keys(), key=lambda x: page_chunks[x], reverse=True)[:3]
        
        # Identify top images by score (Top 4)
        top_images = sorted(image_to_rank, key=lambda x: x['score'], reverse=True)[:4]
        
        # EXECUTE VLM CALLS IN PARALLEL
        visual_findings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Only process if they manual or similarity score > 0.1
            active_images = [img for img in top_images if img['type'] == "manual" or img['score'] > 0.1]
            future_to_img = {executor.submit(self.process_image_node, img, safe_query): img for img in active_images}
            
            for future in concurrent.futures.as_completed(future_to_img):
                result = future.result()
                if result:
                    visual_findings.append(result)

        # 4. Layer 4: Cohesive Synthesis (Llama 4 Parallel Context)
        context_str = "\n\n".join([f"--- [FULL PAGE TEXT - p{pn}] ---\n{full_page_text_map[pn][:4000]}" for pn in top_page_nums])
        visual_str = "\n".join(visual_findings)
        
        prompt = (
            "You are LENS (Multimodal Document Intelligence). Synthesize a PROFESSIONAL ANALYSIS REPORT.\n"
            f"User Query: '{safe_query}'\n\n"
            f"TEXT CONTEXT (Ranked PDF Pages):\n{context_str}\n\n"
            f"VISUAL CONTEXT (Natively reasoned parallel images):\n{visual_str if visual_findings else 'No visual evidence detected.'}\n\n"
            "REPORT RULES (CRITICAL):\n"
            "1. Answer based BOTH on text [pN] and visual [VISUAL_SOURCE] evidence.\n"
            "2. Provide 3 specific sections: 🔍 ANALYSIS SUMMARY | 📖 DETAILED INSIGHTS | ✅ CONCLUSION.\n"
            "3. Be extremely precise. Cite sources explicitly like [p1] or [VISUAL_SOURCE].\n"
            "4. Combine text and image findings into a unified narrative."
        )

        try:
            import requests
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LENS Parallel Synthesis Fail: {e}")
            return f"--- LENS ANALYSIS (Recovery) ---\n\n{context_str[:2000]}\n\n{visual_str}"
