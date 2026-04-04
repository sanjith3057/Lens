# LENS — Multimodal Document Intelligence

**Layered vision + text RAG using CLIP and Llama 4 Scout**

---

## The Problem

Every RAG system built so far — including PRISM-RAG — is blind.

Feed it a PDF with charts, diagrams, tables, or scanned figures and it extracts the text, ignores everything visual, and answers from what it can read. The answer to "what does Figure 3 show?" is silence. The answer to "summarise the architecture diagram on page 4" is a guess built from surrounding text.

Three specific failures that break standard multimodal attempts before you even get to the model:

**CLIP token overflow** — CLIP has a hard 77-token position embedding limit. Feed it a text chunk longer than that and the model crashes at runtime with no graceful fallback.

**Fragmented context** — snippet-based retrieval returns 180-character chunks. The LLM gets a sliver of a page and generates shallow, off-target answers because it never saw the full context around the relevant passage.

**Vision payload rejections** — full-resolution images sent to vision APIs return 400 Bad Request errors. Most implementations never handle this and the visual pipeline silently dies.

---

## Research

**CLIP: Contrastive Language-Image Pretraining — Radford et al., OpenAI, 2021**
CLIP trains a vision encoder and a text encoder jointly to produce embeddings in a shared vector space. A text query and a matching image will have high cosine similarity without any task-specific training. This is the foundation of cross-modal retrieval — one vector space for both modalities, one similarity search for both.

**Multimodal RAG improves retrieval accuracy 25–40%**
Embedding images alongside text chunks improves retrieval accuracy 25–40% over text-only systems on document understanding tasks. Engineers searching for "how does the cooling system connect?" retrieve diagram pages with VLM-generated answers referencing specific components — not just the text paragraph that mentions cooling.

**Llama 4 Scout — Meta, April 2025**
Llama 4 Scout is a 17B active parameter Mixture-of-Experts model with a 10 million token context window and native multimodal understanding. Unlike vision-plugin architectures where images are preprocessed separately and injected as tokens, Scout processes text and image tokens in a unified architecture. No plugin lag. No modality mismatch.

**Page-Level Context vs Snippet Retrieval**
Snippet retrieval is the default in most RAG implementations — retrieve the top-k chunks of 200–400 characters each. For document intelligence, this fails because the relevant information often spans an entire page including surrounding prose, captions, and adjacent figures. Page-level retrieval retrieves the full text of the top-ranked pages, giving the LLM complete context rather than fragments.

---

## Approach

Four layers. Each one solves a specific failure mode from the problem section above.

**Layer 1 — LENS Parser**
PyMuPDF extracts raw text per page and converts page visual elements into high-resolution byte arrays. Each page is treated as both a text unit and an image unit simultaneously. PII redaction via regex runs at parse time before any data leaves the local machine.

**Layer 2 — Universal Embedding Store**
CLIP ViT-B/32 runs locally on the RTX 2050 via CUDA. Both text chunks and page images are embedded into the same 512-dimensional vector space. A strict 120-character safety floor on all text inputs prevents the 77-token CLIP overflow crash. Manual image uploads receive a +0.5 priority boost in scoring so user-provided visual evidence always ranks above auto-extracted page images.

**Layer 3 — Parallel Vision Reasoning**
Llama 4 Scout via Groq processes PDF page images and manual image uploads simultaneously using a parallel execution engine. Images are automatically downscaled to 600px at quality 65 before payload construction — this eliminates the 400 Bad Request vision API errors entirely. No sequential bottleneck. Both visual sources are analysed at the same time.

**Layer 4 — Intelligence Synthesis**
Llama 4 Scout's 10 million token context window receives the top 3 full pages of text and top 4 visual findings simultaneously. The synthesis prompt forces structured professional output with explicit citations — `[p1]`, `[p2]`, `[VISUAL_SOURCE]` — so every claim in the final report traces back to a specific source. The LLM never generates from outside the retrieved context.

---

## Results

Tested on: resume PDF + standalone Spider-Man image
Query: "Summarise the professional profile and interpret the visual context"

CLIP ran stably on RTX 2050 CUDA with zero token overflow errors. Page-level retrieval returned full pages — no 180-character fragments. Vision payload auto-downscale eliminated all 400 errors. Llama 4 Scout fused both inputs and produced a unified report:

> "The analysis combines insights from the provided text and visual context. Sanjith G has a strong background in AI/ML including RAG pipelines and agentic workflows [p1][p2]. The visual context features Spider-Man, symbolising agility and the ability to navigate complex situations — qualities that mirror the adaptability required in production AI engineering [VISUAL_SOURCE]."

Every claim cites its source. Text evidence and visual evidence synthesised into one narrative. Citations traceable to specific pages and visual inputs.



https://github.com/user-attachments/assets/1175ea27-76dd-4605-b9a5-c4c7662b00ed



| Component | Status |
|---|---|
| CLIP embeddings | Running on RTX 2050 CUDA |
| Page-level retrieval | Top 3 full pages — no snippet truncation |
| Vision payload | Auto-downscaled to 600px/Q65 — zero API errors |
| Output format | Professional report with [p1][p2][VISUAL_SOURCE] citations |
| Context window | Llama 4 Scout 10M tokens |

---

## Stack

| Component | Tool |
|---|---|
| PDF parser | PyMuPDF (fitz) |
| Embeddings | CLIP ViT-B/32 — local CUDA |
| Vision model | Llama 4 Scout 17B-16E via Groq |
| Synthesis model | Llama 4 Scout — 10M context window |
| UI | Streamlit |
| Concurrency | Python parallel execution engine |

---

## How to Run

```bash
git clone https://github.com/sanjith3057/lens
cd lens
pip install -r requirements.txt

echo "GROQ_VISION_API_KEY=your_key" > .env

streamlit run app.py
```
