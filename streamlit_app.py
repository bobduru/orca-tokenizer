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
    parts_df = pd.read_csv("./data/parts_manual_labels_v4.csv")
    online_df = pd.read_csv("./data/ford-catalogue/online_catalogue.csv")
    return parts_df, online_df

parts_df, online_catalogue_df = load_data()
paper_spects_dir = Path("./data/ford_paper_spects/")



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
    show_details = st.toggle("Show Detailed Part Descriptions", value=True)
with col2:
    show_online = st.toggle("Show Online Examples", value=False)
    st.session_state.show_online_global = show_online

# Grid settings
cols_per_row = 4

st.markdown("---")

# Custom CSS for card styling
# st.markdown("""
# <style>
# .stColumn{
#     border: 1px solid #e0e0e0;
#     border-radius: 12px;
#     padding: 10px;
#     box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
#     margin-bottom: 20px;
#     background-color: white;
# }
# </style>
# """, unsafe_allow_html=True)

# Display calls in grid
for idx in range(0, len(parts_df), cols_per_row):
    cols = st.columns(cols_per_row)
    
    for col_idx, col in enumerate(cols):
        row_idx = idx + col_idx
        if row_idx >= len(parts_df):
            break
            
        call_data = parts_df.iloc[row_idx]
        
        # If not checked, create empty call_data for display
        if not call_data.get('checked_v4', False):
            display_data = call_data.copy()
            for i in range(1, 6):
                display_data[f'P{i}_tokens'] = ''
            display_data['notes'] = ''
        else:
            display_data = call_data
        
        with col:
            render_call_card(
                display_data, 
                online_catalogue_df, 
                paper_spects_dir,
                show_details=show_details,
                show_online=st.session_state.show_online_global,
                card_idx=row_idx
            )

st.markdown("---")
st.caption(f"Total calls: {len(parts_df)}")
