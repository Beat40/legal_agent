
import streamlit as st
import pandas as pd
import os
import torch
from io import BytesIO
import legal_lib
import time
import importlib

# Force reload of legal_lib to catch changes without manual restart
importlib.reload(legal_lib)

# --- Page Config ---
st.set_page_config(
    page_title="Agentic Legal Contract Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Custom Styling ---
st.markdown("""
<style>
    .highlight {
        background-color: #ffe066;
        padding: 2px 4px;
        border-radius: 4px;
        border-bottom: 2px solid #ffd700;
        cursor: help; /* change cursor to question mark on hover */
    }
    .highlight:hover {
        background-color: #ffd700;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
    .evidence-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #4e8cff;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0.0
if 'timings' not in st.session_state:
    st.session_state.timings = {"BERT": 0.0, "Labeling": 0.0, "Reasoning": 0.0}

# --- Caching Resources ---
@st.cache_resource
def get_models():
    return legal_lib.load_models()

@st.cache_resource
def get_data():
    return legal_lib.load_data()

@st.cache_resource
def get_llm():
    return legal_lib.init_llm()

# --- Load Resources ---
try:
    qa_model, qa_tokenizer, embedder = get_models()
    index, chunks_df = get_data()
    llm = get_llm()
    # Cache CUAD questions once
    cached_questions = legal_lib.cache_cuad_questions(qa_tokenizer, legal_lib.CUAD_QA_SCHEMA)
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    # Branding
    if os.path.exists("assets/bits_logo.png"):
        st.image("assets/bits_logo.png", use_container_width=True)
    else:
        st.header("BITS Pilani")
        st.caption("Work Integrated Learning Programmes")

    st.title("Settings")
    
    uploaded_file = st.file_uploader("Upload Contract (TXT)", type="txt")
    
    process_btn = st.button("Run Analysis", type="primary", disabled=not uploaded_file)
    
    st.markdown("---")
    st.markdown("### Hardware Status")
    hw_placeholder = st.empty()

    def update_hw_display(placeholder):
        cuda_available = torch.cuda.is_available()
        with placeholder.container():
            st.write(f"**CUDA Available**: {'✅' if cuda_available else '❌'}")
            if cuda_available:
                try:
                    current_device = torch.cuda.current_device()
                    vram_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                    vram_used = torch.cuda.memory_allocated(current_device) / (1024**3)
                    st.write(f"**Device**: {torch.cuda.get_device_name(current_device)}")
                    st.write(f"**Model on**: {next(qa_model.parameters()).device}")
                    st.progress(min(vram_used / vram_total, 1.0), text=f"VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB")
                except Exception:
                    st.write(f"**Device**: GPU Detected")
    
    update_hw_display(hw_placeholder)

    st.markdown("---")
    st.markdown("### Agentic Pipeline Status")
    status_container = st.empty()

# --- Main Logic ---

def run_pipeline(text, hw_update_fn, hw_placeholder):
    start_time = time.time()
    st.session_state.timings = {"BERT": 0.0, "Labeling": 0.0, "Reasoning": 0.0}
    progress_bar = st.progress(0)
    status_container.info("Chunking Contract...")
    
    # 1. Chunking
    chunks = legal_lib.chunk_contract(text, qa_tokenizer)
    progress_bar.progress(10)
    
    # 2. Extraction (BERT)
    status_container.info("Agent 2: CUAD QA Tagger (BERT Extraction)...")
    hw_update_fn(hw_placeholder) # Show VRAM before BERT
    t_start = time.time()
    # Pass callback to update VRAM during the loop
    all_results, _ = legal_lib.run_contract_analysis_sequential(
        chunks, qa_model, qa_tokenizer, cached_questions, 
        callback=lambda: hw_update_fn(hw_placeholder)
    )
    st.session_state.timings["BERT"] = time.time() - t_start
    hw_update_fn(hw_placeholder) # Show VRAM after BERT
    
    # Filter unique spans
    unique_spans = list(set([r["span"] for r in all_results if r["confidence"] > 7.5]))
    progress_bar.progress(40)
    
    if not unique_spans:
        status_container.warning("No high-confidence clauses found.")
        return []

    # 3. LLM Labeling
    status_container.info("Agent 2: Refining Labels with GenAI...")
    t_start = time.time()
    label_map = legal_lib.label_spans_with_llm_parallel(unique_spans, llm)
    st.session_state.timings["Labeling"] = time.time() - t_start
    progress_bar.progress(60)
    
    final_data = []
    
    # 4. Retrieval & Reasoning Loop (Parallelized)
    status_container.info("Agents 3 & 4: Retrieval and Reasoning (Parallel)...")
    t_start = time.time()
    total_spans = len(unique_spans)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx, span in enumerate(unique_spans):
            label = label_map.get(span, "General")
            futures.append(executor.submit(
                legal_lib.process_span_full, 
                idx, span, label, embedder, index, chunks_df, llm
            ))
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            final_data.append(result)
            # Update progress
            current_progress = 60 + int(((i + 1) / total_spans) * 40)
            progress_bar.progress(min(current_progress, 100))
    
    st.session_state.timings["Reasoning"] = time.time() - t_start
    # Re-sort final_data by ID to keep order if consistency is preferred
    final_data = sorted(final_data, key=lambda x: x['id'])

    progress_bar.progress(100)
    st.session_state.total_time = time.time() - start_time
    status_container.success(f"Analysis Complete in {st.session_state.total_time:.2f}s!")
    time.sleep(1)
    status_container.empty()
    hw_update_fn(hw_placeholder) # Final VRAM check
    return final_data

if process_btn and uploaded_file:
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    st.session_state.contract_text = text
    with st.spinner("Agents are working..."):
        results = run_pipeline(text, update_hw_display, hw_placeholder)
        st.session_state.analysis_results = results
        st.session_state.processing_complete = True

# --- Visualization ---

if st.session_state.processing_complete:
    
    # Dashboard Stats
    col1, col2, col3 = st.columns(3)
    total_clauses = len(st.session_state.analysis_results)
    reviewed = sum(1 for r in st.session_state.analysis_results if r['status'] != "Review Pending")
    with col1:
        st.metric("Clauses Extracted", total_clauses)
    with col2:
        st.metric("Clauses Reviewed", reviewed)
    with col3:
        st.metric("Processing Time", f"{st.session_state.total_time:.2f}s")
    
    # Detailed Timings
    with st.expander("⏱️ Detailed Timing Breakdown"):
        t_cols = st.columns(3)
        t_cols[0].metric("BERT Extraction", f"{st.session_state.timings['BERT']:.1f}s")
        t_cols[1].metric("LLM Labeling", f"{st.session_state.timings['Labeling']:.1f}s")
        t_cols[2].metric("Retrieval/Reasoning", f"{st.session_state.timings['Reasoning']:.1f}s")
    
    st.markdown("---")
    
    # Layout: Text on Left (scrollable), Review Panel on Right
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.subheader("Contract View")
        # Highlight Logic
        # We replace the span text with HTML spans. 
        # To avoid replacement conflicts (nested spans), we sort by length desc or process carefully.
        # Simple approach: Replace unique strings.
        
        annotated_html = st.session_state.contract_text
        # Sort by length desc to prevent replacing sub-strings first
        sorted_results = sorted(st.session_state.analysis_results, key=lambda x: len(x['span']), reverse=True)
        
        for item in sorted_results:
            span_text = item['span']
            label = item['label']
            # Simple tooltip with Label
            tooltip = f"Label: {label}"
            replacement = f'<span class="highlight" title="{tooltip}">{span_text}</span>'
            # Use strict replace to avoid messing up HTML
            annotated_html = annotated_html.replace(span_text, replacement)
            
        st.markdown(f"""
        <div style="height: 600px; overflow-y: scroll; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: white; font-family: monospace; white-space: pre-wrap;">
        {annotated_html}
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("Review Panel")
        
        if not st.session_state.analysis_results:
            st.info("No clauses to review.")
        
        for idx, item in enumerate(st.session_state.analysis_results):
            with st.expander(f"{item['label']} - {item['span'][:50]}...", expanded=False):
                st.markdown(f"**Clause:**\n> {item['span']}")
                
                st.markdown(f"**Analysis:**\n{item['analysis']}")
                
                st.markdown("**Evidence:**")
                # Show top 3 evidence items
                for ev in item['evidence'][:3]:
                    # Using a dialog for full text
                    # We create a button for each evidence piece
                    col_ev1, col_ev2 = st.columns([3, 1])
                    with col_ev1:
                        st.caption(f"ID: {ev['celex_id']} (Score: {ev['score']:.2f})")
                    with col_ev2:
                        # Use a unique key for the popover
                        with st.popover("View Text"):
                            st.markdown(f"### {ev['title']}")
                            st.markdown(f"**ID**: {ev['celex_id']}")
                            st.markdown("---")
                            st.markdown(ev['text'])
                
                st.markdown("---")
                # Feedback Controls
                c_status, c_comment = st.columns([1, 2])
                
                with c_status:
                    status = st.radio(
                        "Action", 
                        ["Review Pending", "Accept", "Reject"], 
                        key=f"status_{idx}",
                        index=["Review Pending", "Accept", "Reject"].index(item['status'])
                    )
                    # Update state
                    st.session_state.analysis_results[idx]['status'] = status
                    
                with c_comment:
                    comment = st.text_area(
                        "Comments", 
                        value=item['comment'], 
                        key=f"comment_{idx}"
                    )
                    st.session_state.analysis_results[idx]['comment'] = comment

    # --- Export ---
    st.markdown("---")
    
    # Prepare DataFrame for export
    export_data = []
    for item in st.session_state.analysis_results:
        export_row = {
            "Clause": item['span'],
            "Label": item['label'],
            "Analysis": item['analysis'],
            "Review Status": item['status'],
            "User Comments": item['comment'],
            "Top Evidence ID": item['evidence'][0]['celex_id'] if item['evidence'] else "N/A"
        }
        export_data.append(export_row)
        
    df_export = pd.DataFrame(export_data)
    
    # Excel Download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Review')
    processed_data = output.getvalue()
    
    st.download_button(
        label="Download Report (Excel)",
        data=processed_data,
        file_name="legal_analysis_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif not uploaded_file:
    st.info("👈 Please upload a contract in the sidebar to begin.")
