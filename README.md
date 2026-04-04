# 👁️ LENS: Multimodal Document Intelligence

**LENS** is an advanced Multimodal RAG (Retrieval-Augmented Generation) system built to extract, rank, and synthesize insights from complex documents (PDFs) and manual image uploads. By leveraging a natively multimodal **Llama 4 Scout** core and local **CLIP** embeddings, LENS provides high-precision analysis where text and visual evidence are fused into a single professional narrative.

---

## 🚩 The Problem
Traditional RAG systems are "blind." They treat PDFs as flat text strings, losing critical visual context (charts, diagrams, signatures, and page layouts). Furthermore, early multimodal attempts often suffer from:
- **CLIP Token Overflows**: Strict 77-token limits causing runtime crashes.
- **Fragmented Context**: Retrieving tiny snippets (e.g., 180 chars) resulting in shallow, "not to the mark" answers.
- **Vision Payload Rejections**: API errors (400 Bad Request) due to oversized or improperly formatted image data.

## 🚀 The LENS Approach (Solution)
LENS solves these by implementing a **Layered Multimodal Architecture** with three core innovations:
1. **Natively Multimodal MoE**: Upgraded to **Llama 4 Scout 17B-16E**, a natively multimodal Mixture-of-Experts model that understands text and image tokens in a unified architecture (no vision-plugin lag).
2. **Page-Level Retrieval**: Instead of snippet-based context, LENS identifies the top-ranked sections and retrieves the **entire page text**, ensuring the LLM has deep context for professional reporting.
3. **Concurrent Reasoning**: Uses a parallel execution engine to analyze PDF images and manual uploads simultaneously, eliminating sequential bottlenecking.

---

## 🏗️ Technical Architecture (The 4 Layers)

### Layer 1: The LENS Parser (`src/parser.py`)
- **Engine**: Powered by `PyMuPDF` (fitz).
- **Function**: Extracts raw text and converts page visual elements into high-res byte arrays. 
- **Privacy**: Implements regex-based PII redaction for secure analysis.

### Layer 2: Universal Embedding Store (`src/embedding.py`)
- **Model**: Local **CLIP (ViT-B-32)** running on the local GPU (RTX 2050/CUDA).
- **Hardening**: Implements a strict **120-character safety floor** to prevent 77-token CLIP position embedding crashes.
- **Ranking**: Unifies text chunks and image features into a single vector space for semantic similarity ranking.

### Layer 3: Parallel Vision reasoning (`src/vlq.py`)
- **Model**: Llama-4-Scout (Groq).
- **Hardening**: Automatically downscales images to **600px/Quality 65** for rapid, stable API payloads.
- **Concurrency**: Parallel analysis of multiple visual sources (PDF diagrams + Manual uploads) simultaneously.

### Layer 4: Intelligence Synthesis (`src/answer.py`)
- **Synthesis Brain**: Llama-4-Scout (10M token context window).
- **Output**: Generates formatted professional reports with specific citations (`[p1]`, `[VISUAL_SOURCE]`).

---

## 🛠️ Working Mechanics
1. **Ingestion**: User uploads a PDF and/or images via the Streamlit dashboard.
2. **Vector Mapping**: CLIP generates embeddings for the query and all document components.
3. **Multimodal Ranking**: LENS scores every text chunk and image. Manual images receive a **+0.5 priority boost**.
4. **Context Assembly**: The top 3 most relevant full pages are gathered along with the top 4 visual findings.
5. **Final Inference**: Llama 4 Scout synthesizes the final report using the 10M context window to ensure "to the mark" precision.

---

## 📥 Installation

### 1. Requirements
- Python 3.10+
- NVIDIA GPU (Recommended for CLIP)
- Groq API Key

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/LENS.git
cd LENS

# Install dependencies
pip install -r requirements.txt

# Configure Environment
# Create a .env file:
GROQ_VISION_API_KEY=your_key_here
```

### 3. Launch
```bash
streamlit run app.py
```

---

## 🔍 Features at a Glance
- ✅ **Llama 4 Scout Upgrade**: Natively multimodal reasoning.
- ✅ **Concurrent Vision**: Parallel processing of multiple images.
- ✅ **Page-Level Context**: Deep document understanding.
- ✅ **CLIP Protection**: Recursive chunking/truncation (100% stable).
- ✅ **High Contrast UI**: Professional dark-mode dashboard.

---

