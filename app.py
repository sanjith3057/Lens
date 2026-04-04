import streamlit as st
import os
import io
from PIL import Image
from src.answer import AnswerGenerator
from src.__init__ import SecureConfig, logger
import tempfile

# Page config
st.set_page_config(
    page_title="LENS | Multimodal Document Intelligence",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for premium high-contrast look
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #238636; color: white; border: none; font-weight: bold;
    }
    .report-box {
        background-color: #161b22; padding: 25px; border-radius: 12px;
        border: 1px solid #30363d; font-family: 'Inter', sans-serif;
        color: #f8f9fa !important; line-height: 1.6;
    }
    .report-box h1, .report-box h2, .report-box h3, .report-box p, .report-box li {
        color: #f8f9fa !important;
    }
    .citation-tag {
        background-color: #388bfd; color: white; padding: 2px 6px;
        border-radius: 4px; font-size: 0.85em; font-weight: bold;
    }
    .status-badge {
        background-color: #21262d; border: 1px solid #30363d; padding: 10px;
        border-radius: 8px; color: #8b949e; font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_generator():
    # Use st.status to show model loading feedback
    with st.status("🚀 Loading Local LENS Intelligence...", expanded=True) as status:
        st.write("Checking GPU (RTX 2050) compatibility...")
        gen = AnswerGenerator()
        st.write("CLIP-ViT-B-32 model loaded.")
        status.update(label="✅ LENS Ready!", state="complete", expanded=False)
    return gen

def main():
    st.title("🔍 LENS — Multimodal Document Explorer")
    st.markdown("### Professional Document Intelligence (Text & Vision)")
    
    # Sidebar: Status & Config
    with st.sidebar:
        st.header("⚙️ LENS 4-Layer Status")
        # Check Groq Key
        try:
            SecureConfig.get_api_key('GROQ_VISION_API_KEY')
            st.success("✅ Layer 3 (Vision): Connected")
        except:
            st.error("❌ Groq API Key Missing")
        
        # Check Local Model status
        st.info("🧠 Layer 2 (Embedding): CLIP-Local (CUDA)")
        st.divider()
        st.markdown("Implemented strictly to **Day 3** specifications.")

    # Ingestion Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📥 Ingestion")
        uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
        uploaded_images = st.file_uploader("Upload Standalone Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        query = st.text_area("Analysis Query", value="Summarize the key findings from both text and visuals.", height=150)
        
        analyze_btn = st.button("🚀 Analyze All Modalities")

    with col2:
        st.subheader("📊 LENS Comprehensive Analysis")
        
        if analyze_btn:
            if not uploaded_pdf and not uploaded_images:
                st.warning("Please upload a PDF or an Image first.")
                return

            with st.spinner("LENS is analyzing modalities (Text + Vision)..."):
                try:
                    # Collect manual images
                    manual_image_bytes = []
                    if uploaded_images:
                        for img_file in uploaded_images:
                            manual_image_bytes.append(img_file.getvalue())
                    
                    # Handle PDF if exists
                    pdf_path = None
                    if uploaded_pdf:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                            tmp_pdf.write(uploaded_pdf.getvalue())
                            pdf_path = tmp_pdf.name
                    
                    generator = get_generator()
                    report = generator.generate(pdf_path, query, manual_images=manual_image_bytes)
                    
                    # Display the report with highlighting for citations
                    formatted_report = report.replace("[TEXT-", '<span class="citation-tag">[TEXT-').replace("[IMG-", '<span class="citation-tag">[IMG-').replace("]", "]</span>")
                    st.markdown(f'<div class="report-box">{formatted_report}</div>', unsafe_allow_html=True)
                    
                    # Log retrieval for visibility
                    st.success("Multimodal synthesis complete.")
                    
                    # Cleanup
                    if pdf_path:
                        os.unlink(pdf_path)
                        
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")
                    logger.error(f"LENS UI Error: {e}")
        else:
            st.info("Contextual results will appear here after ingestion.")

if __name__ == "__main__":
    main()
