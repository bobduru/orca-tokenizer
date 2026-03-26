import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os


sys.path.append(str(Path(__file__).parent.parent))  # UI/ (for Streamlit pages)

from token_library.call_query import build_index, search_calls, load_index_from_csv
from components import render_call_card
from ai_classification.ai_query import annotate_spectrogram

# Page config
st.set_page_config(
    page_title="Call Query",
    page_icon="🔍",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
   
    index, DB, calls_df = load_index_from_csv("./data/ford-catalogue/merged_online_catalogue_annotated_parsed.csv")
    return index, DB, calls_df

index, DB, calls_df = load_data()
paper_spects_dir = Path("./data/ford_paper_spects/")

# Initialize session state for AI results
if 'ai_annotation' not in st.session_state:
    st.session_state.ai_annotation = None
if 'ai_prediction' not in st.session_state:
    st.session_state.ai_prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'last_uploaded_name' not in st.session_state:
    st.session_state.last_uploaded_name = None

# Initialize session state for carousel (for components)
if 'current_online_idx' not in st.session_state or not isinstance(st.session_state.current_online_idx, dict):
    st.session_state.current_online_idx = {}

# Title
st.title("🔍 Call Query")
st.markdown("Search for orca calls using tokenized descriptions")

# Spectrogram upload and AI analysis
st.markdown("### Upload Spectrogram for AI Analysis (Optional)")
uploaded_file = st.file_uploader(
    "Drag and drop a spectrogram image",
    type=["png", "jpg", "jpeg"],
    help="Upload a spectrogram to get AI-generated token suggestions"
)

# Show AI Thoughts button if file is uploaded
if uploaded_file is not None:
    # Check if this is a new file - clear AI results if so
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.ai_annotation = None
        st.session_state.ai_prediction = None
        st.session_state.last_uploaded_name = uploaded_file.name
    
    # Store uploaded image in session state
    st.session_state.uploaded_image = uploaded_file.getvalue()
    
    col_img, col_ai = st.columns([1, 2])
    
    with col_img:
        st.image(uploaded_file, caption="Uploaded Spectrogram", width=300)
    
    with col_ai:
        if st.button("🤖 AI Thoughts", type="primary", use_container_width=True):
            with st.spinner("Analyzing spectrogram with AI..."):
                tmp_path = None
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Set OpenAI API key from Streamlit secrets
                    import openai
                    if "OPENAI_API_KEY" in st.secrets:
                        openai.api_key = st.secrets["OPENAI_API_KEY"]
                        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                    
                    # Call AI annotation (display_image=False for Streamlit)
                    annotation, prediction = annotate_spectrogram(tmp_path, display_image=False)
                    
                    # Store results in session state
                    st.session_state.ai_annotation = annotation
                    st.session_state.ai_prediction = prediction
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
                    st.exception(e)
                finally:
                    # Clean up temp file
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        # Display AI results from session state (persists across reruns)
        if st.session_state.ai_annotation is not None:
            st.markdown(f"**Predicted Tokens:**")
            st.code(st.session_state.ai_annotation, language=None)
            st.markdown(f"**Predicted Call: {st.session_state.ai_prediction}**")

elif st.session_state.uploaded_image is not None:
    # Show previous image and results if they exist
    col_img, col_ai = st.columns([1, 2])
    
    with col_img:
        st.image(st.session_state.uploaded_image, caption="Uploaded Spectrogram", width=300)
    
    with col_ai:
        if st.session_state.ai_annotation is not None:
            st.markdown(f"**Predicted Tokens:**")
            st.code(st.session_state.ai_annotation, language=None)
            st.markdown(f"**Predicted Call: {st.session_state.ai_prediction}**")
else:
    # No file uploaded - clear the last uploaded name
    st.session_state.last_uploaded_name = None

st.markdown("---")

# Search input
query = st.text_input(
    "Enter search query",
    placeholder="e.g., UP FAST PEAK FLAT, SQUIGGLE, etc.",
    help="Use tokens like: UP, DOWN, FLAT, PEAK, VALLEY, SQUIGGLE, TIGHT, BLUR, etc."
)

# Top controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    show_details = st.toggle("Show Detailed Part Descriptions", value=True, key="query_show_details")
with col2:
    show_online = st.toggle("Show Online Examples", value=False, key="query_show_online")
with col3:
    topk = st.slider("Number of results", min_value=1, max_value=20, value=8, key="query_topk")

st.markdown("---")

# Search button and results
if query:
    with st.spinner("Searching..."):
        try:
            # Perform search
            results_df = search_calls(query, index, topk=topk)
            
            if len(results_df) == 0:
                st.warning("No results found")
            else:
                st.success(f"Found {len(results_df)} results")
                
                # Display results in grid
                cols_per_row = 4
                for idx in range(0, len(results_df), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for col_idx, col in enumerate(cols):
                        result_idx = idx + col_idx
                        if result_idx >= len(results_df):
                            break
                        
                        result = results_df.iloc[result_idx]
                        filename = result['Call']
                        score = result['Score']
                        matched = result['Matched']
                        
                        # Get all rows for this call type
                        call_rows = calls_df[calls_df['filename'] == filename]

                        call_name = call_rows['call_type'].iloc[0]

                        
                        if len(call_rows) > 0:
                            with col:
                                render_call_card(
                                    call_type=call_name,
                                    call_rows=call_rows,
                                    paper_spects_dir=paper_spects_dir,
                                    show_details=show_details,
                                    show_online=show_online,
                                    score=score,
                                    matched=matched,
                                    card_idx=result_idx
                                )
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Enter a search query above to find matching calls")

st.markdown("---")
st.caption(f"Total indexed call types: {len(index)}")
