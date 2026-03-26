import streamlit as st
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Label Orca Calls",
    page_icon="🏷️",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    parts_df = pd.read_csv("./data/parts_manual_labels_v4.csv")
    online_df = pd.read_csv("./data/ford-catalogue/online_catalogue.csv")
    return parts_df, online_df

def save_to_v4(df):
    """Save dataframe to v4"""
    df.to_csv("./data/parts_manual_labels_v4.csv", index=False)
    st.cache_data.clear()  # Clear cache to reload data

# Load data
parts_df, online_catalogue_df = load_data()
paper_spects_dir = Path("./data/ford_paper_spects/")

# Create navigation
call_names = parts_df['Call'].tolist()

# Get current call from query params
if 'call_name' in st.query_params:
    call_name = st.query_params['call_name']
else:
    # Default to first call if no query param
    call_name = parts_df.iloc[0]['Call']
    st.query_params['call_name'] = call_name

# Initialize session state for online carousel (per call)
if 'online_carousel' not in st.session_state:
    st.session_state.online_carousel = {}
if call_name not in st.session_state.online_carousel:
    st.session_state.online_carousel[call_name] = 0

# Get current call data
try:
    call_data = parts_df[parts_df['Call'] == call_name].iloc[0]
    current_idx = parts_df[parts_df['Call'] == call_name].index[0]
except IndexError:
    # If call not found, default to first
    call_name = parts_df.iloc[0]['Call']
    st.query_params['call_name'] = call_name
    call_data = parts_df.iloc[0]
    current_idx = 0

# Sidebar navigation
with st.sidebar:
    st.markdown("## Calls")
    for idx, cn in enumerate(call_names):
        checked = "✓" if parts_df.iloc[idx]['checked_v4'] else "○"
        label = f"{checked} {cn}"
        if st.button(label, key=f"nav_{cn}", use_container_width=True, 
                    type="primary" if cn == call_name else "secondary"):
            st.query_params['call_name'] = cn
            # Initialize carousel for new call if needed
            if cn not in st.session_state.online_carousel:
                st.session_state.online_carousel[cn] = 0
            st.rerun()

# Title
checked = "✓" if call_data.get('checked_v4', False) else "○"
st.title(f"🏷️ Label: {call_name} {checked}")
st.caption(f"Call {current_idx + 1} of {len(parts_df)}")

# Main layout: Spectrograms on left (1/3), Labeling form on right (2/3)
left_col, right_col = st.columns([1, 2])

with left_col:
    # Paper spectrogram
    st.markdown("**Paper Spectrogram**")
    paper_spect = paper_spects_dir / f"{call_name}_paper_spect.png"
    if paper_spect.exists():
        st.image(str(paper_spect), use_container_width=True)
    else:
        st.warning("Not found")
    
    st.markdown("---")
    
    # Online examples
    st.markdown("**Online Examples**")
    online_rows = online_catalogue_df[online_catalogue_df['call_type'] == call_name]
    
    if len(online_rows) > 0:
        current_online_idx = st.session_state.online_carousel[call_name]
        if current_online_idx >= len(online_rows):
            current_online_idx = 0
            st.session_state.online_carousel[call_name] = 0
        
        current_row = online_rows.iloc[current_online_idx]
        spect_path = Path(current_row['spect_fp'])
        audio_path = current_row.get('audio_fp', '')
        
        if spect_path.exists():
            clan_info = current_row.get('clan', 'N/A')
            pod_info = current_row.get('pod', 'N/A')
            st.image(str(spect_path), use_container_width=True)
            st.caption(f"{current_online_idx + 1}/{len(online_rows)} | Clan: {clan_info} | Pod: {pod_info}")
            
            # Audio player
            if audio_path and Path(audio_path).exists():
                st.audio(str(audio_path))
            
            # Navigation
            if len(online_rows) > 1:
                nav_cols = st.columns([1, 1])
                with nav_cols[0]:
                    if st.button("◀", use_container_width=True, key="prev_online"):
                        st.session_state.online_carousel[call_name] = (current_online_idx - 1) % len(online_rows)
                        st.rerun()
                with nav_cols[1]:
                    if st.button("▶", use_container_width=True, key="next_online"):
                        st.session_state.online_carousel[call_name] = (current_online_idx + 1) % len(online_rows)
                        st.rerun()
    else:
        st.info("No online examples")

with right_col:
    # Get number of parts
    num_parts = int(call_data.get('Parts', 0)) if pd.notna(call_data.get('Parts')) else 0
    
    # Create form data dictionary
    form_data = {}
    
    if num_parts > 0:
        # Create columns for each part
        st.markdown("**Part Labels**")
        
        # Header row - show P{i}_old
        header_cols = st.columns(num_parts)
        for i, col in enumerate(header_cols):
            with col:
                original_label = call_data.get(f'P{i+1}_old', '')
                st.markdown(f"**P{i+1}**")
                st.caption(f"{original_label if pd.notna(original_label) else 'N/A'}")
        
        # Tokens row
        st.markdown("**Tokens:**")
        tokens_cols = st.columns(num_parts)
        for i, col in enumerate(tokens_cols):
            part_num = i + 1
            with col:
                current_tokens = call_data.get(f'P{part_num}_tokens', '')
                widget_key = f"tokens_p{part_num}_{call_name}"
                form_data[f'P{part_num}_tokens'] = st.text_input(
                    "Tokens",
                    value=current_tokens if pd.notna(current_tokens) else '',
                    key=widget_key,
                    label_visibility="collapsed"
                )
        
        # Checkboxes
        st.markdown("**Always present:**")
        always_cols = st.columns(num_parts)
        for i, col in enumerate(always_cols):
            part_num = i + 1
            with col:
                current_always = call_data.get(f'P{part_num}_always_present', False)
                widget_key = f"always_p{part_num}_{call_name}"
                form_data[f'P{part_num}_always_present'] = st.checkbox(
                    "Always",
                    value=bool(current_always),
                    key=widget_key,
                    label_visibility="collapsed"
                )
        
        st.markdown("**Strong variation:**")
        variation_cols = st.columns(num_parts)
        for i, col in enumerate(variation_cols):
            part_num = i + 1
            with col:
                current_variation = call_data.get(f'P{part_num}_strong_variation', False)
                widget_key = f"variation_p{part_num}_{call_name}"
                form_data[f'P{part_num}_strong_variation'] = st.checkbox(
                    "Variation",
                    value=bool(current_variation),
                    key=widget_key,
                    label_visibility="collapsed"
                )
    
    st.markdown("---")
    
    # Notes section
    original_notes = call_data.get('old_notes', '')
    # if pd.notna(original_notes) and original_notes:
    st.markdown(f"**Original Notes:** {original_notes}")
    
    st.markdown("**Notes:**")
    current_notes = call_data.get('notes', '')
    widget_key = f"new_notes_{call_name}"
    form_data['notes'] = st.text_area(
        "Notes",
        value=current_notes if pd.notna(current_notes) else '',
        height=80,
        key=widget_key,
        label_visibility="collapsed"
    )

st.markdown("---")

# Navigation and Save
def save_current():
    """Save current call data"""
    # Update dataframe with form data
    for key, value in form_data.items():
        parts_df.at[current_idx, key] = value
    
    # Mark as checked
    parts_df.at[current_idx, 'checked_v4'] = True
    
    # Save to v4
    save_to_v4(parts_df)

def navigate_to_call(direction):
    """Navigate to next or previous call"""
    if direction == 'next':
        new_idx = (current_idx + 1) % len(parts_df)
    else:  # previous
        new_idx = (current_idx - 1) % len(parts_df)
    
    new_call_name = parts_df.iloc[new_idx]['Call']
    st.query_params['call_name'] = new_call_name
    # Initialize carousel for new call if needed
    if new_call_name not in st.session_state.online_carousel:
        st.session_state.online_carousel[new_call_name] = 0
    st.rerun()

# Bottom navigation
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    if st.button("◀ Previous Call", use_container_width=True, type="secondary"):
        save_current()
        navigate_to_call('previous')

with nav_col2:
    if st.button("💾 Save", use_container_width=True, type="primary"):
        save_current()
        st.success(f"✓ Saved labels for {call_name}")
        st.rerun()

with nav_col3:
    if st.button("Next Call ▶", use_container_width=True, type="secondary"):
        save_current()
        navigate_to_call('next')
