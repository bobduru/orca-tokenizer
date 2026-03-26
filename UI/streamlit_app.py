import streamlit as st
import pandas as pd
from pathlib import Path
from components import render_call_card

# Page config
st.set_page_config(
    page_title="Orca Call Catalogue",
    page_icon="🐋",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/ford-catalogue/merged_online_catalogue_annotated_parsed.csv")
    return df

catalogue_df = load_data()
paper_spects_dir = Path("./data/ford_paper_spects/")

# Get unique call types (preserving original order)
call_types = catalogue_df['call_type'].unique()

# Initialize session state for image carousel and show online toggle
if 'current_online_idx' not in st.session_state or not isinstance(st.session_state.current_online_idx, dict):
    st.session_state.current_online_idx = {}
if 'show_online_global' not in st.session_state:
    st.session_state.show_online_global = False

# Title and controls
st.title("🐋 Orca Call Catalogue - Overview")
st.markdown("Navigate to **Label Calls** page from the sidebar to start labeling →")
st.markdown("---")

# Global controls
col1, col2 = st.columns(2)
with col1:
    show_details = st.toggle("Show Detailed Part Descriptions", value=False)
with col2:
    show_online = st.toggle("Show Online Examples", value=True)
    st.session_state.show_online_global = show_online

# Grid settings
cols_per_row = 4

st.markdown("---")

# Display calls in grid
for idx in range(0, len(call_types), cols_per_row):
    cols = st.columns(cols_per_row)
    
    for col_idx, col in enumerate(cols):
        row_idx = idx + col_idx
        if row_idx >= len(call_types):
            break
            
        call_type = call_types[row_idx]
        call_rows = catalogue_df[catalogue_df['call_type'] == call_type]
        
        with col:
            render_call_card(
                call_type,
                call_rows,
                paper_spects_dir,
                show_details=show_details,
                show_online=st.session_state.show_online_global,
                card_idx=row_idx
            )

st.markdown("---")
st.caption(f"Total call types: {len(call_types)} | Total examples: {len(catalogue_df)}")
