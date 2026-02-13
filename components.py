import streamlit as st
import pandas as pd
from pathlib import Path


def render_call_card(call_data, online_catalogue_df, paper_spects_dir, 
                    show_details=True, show_online=False, score=None, matched=None, card_idx=None):
    """
    Render a call card with spectrogram and metadata
    
    Args:
        call_data: Series with call data
        online_catalogue_df: DataFrame with online catalogue
        paper_spects_dir: Path to paper spectrograms
        show_details: Whether to show part descriptions
        show_online: Whether to show online examples
        score: Optional score to display (for search results)
        matched: Optional matched tokens string to display
        card_idx: Optional unique index for this card (for stable button keys)
    """
    call_name = call_data['Call']
    
    # Generate stable key suffix
    if card_idx is not None:
        key_suffix = f"{call_name}_{card_idx}"
    else:
        key_suffix = call_name
    
    # Get online catalogue rows for this call
    online_rows = online_catalogue_df[online_catalogue_df['call_type'] == call_name]
    
    # Initialize state for online carousel
    if 'current_online_idx' not in st.session_state or not isinstance(st.session_state.current_online_idx, dict):
        st.session_state.current_online_idx = {}
    if call_name not in st.session_state.current_online_idx:
        st.session_state.current_online_idx[call_name] = 0
    
    with st.container(border=True):
        # Display call name as clickable link
        if st.button(f"**{call_name}**", key=f"nav_{key_suffix}", use_container_width=True):
            # Set query param and navigate to label page
            st.query_params['call_name'] = call_name
            st.switch_page("pages/1_Label_Calls.py")
        
        # Show score and matched tokens if provided (for search results)
        if score is not None:
            st.caption(f"Score: {score:.2f}")
        if matched:
            st.caption(f"Matched: {matched}")
        
        # Show metadata
        clans = call_data.get('clans', '')
        pods = call_data.get('pods', '')
        st.caption(f"Clans: {clans} | Pods: {pods}")
        
        # Display paper spectrogram
        paper_spect = paper_spects_dir / f"{call_name}_paper_spect.png"
        if paper_spect.exists():
            st.image(str(paper_spect), use_container_width=True)
        else:
            st.warning(f"Paper spect not found")
        
        # Show online examples if toggled
        if show_online and len(online_rows) > 0:
            current_online_idx = st.session_state.current_online_idx[call_name]
            
            # Ensure index is within bounds
            if current_online_idx >= len(online_rows):
                current_online_idx = 0
                st.session_state.current_online_idx[call_name] = 0
            
            # Get current online row
            current_row = online_rows.iloc[current_online_idx]
            spect_path = Path(current_row['spect_fp'])
            audio_path = current_row.get('audio_fp', '')
            
            st.markdown("---")
            
            # Display online spectrogram
            if spect_path.exists():
                clan_info = current_row.get('clan', 'N/A')
                pod_info = current_row.get('pod', 'N/A')
                st.image(str(spect_path), use_container_width=True, 
                        caption=f"Example {current_online_idx+1} / {len(online_rows)} | Clan: {clan_info} | Pod: {pod_info}")
                
                # Audio player
                if audio_path and Path(audio_path).exists():
                    st.audio(str(audio_path))
                
                # Navigation with clan/pod info
                if len(online_rows) > 1:
                    nav_cols = st.columns([1, 1])
                    with nav_cols[0]:
                        if st.button("◀", key=f"prev_{key_suffix}", use_container_width=True):
                            st.session_state.current_online_idx[call_name] = (current_online_idx - 1) % len(online_rows)
                            st.rerun()
                    with nav_cols[1]:
                        if st.button("▶", key=f"next_{key_suffix}", use_container_width=True):
                            st.session_state.current_online_idx[call_name] = (current_online_idx + 1) % len(online_rows)
                            st.rerun()
            else:
                st.warning(f"Online spect not found: {spect_path}")
        
        # Show details if toggle is on
        if show_details:
            st.markdown("**Part Tokens:**")
            
            # Display parts in a compact format
            parts_desc = []
            for i in range(1, 6):  # P1 to P5
                part_col = f'P{i}_tokens'
                if part_col in call_data and pd.notna(call_data[part_col]) and call_data[part_col] != '':
                    parts_desc.append(f"**P{i}:** {call_data[part_col]}")
            
            if parts_desc:
                st.markdown("<br>".join(parts_desc), unsafe_allow_html=True)
            
            # Display notes
            if pd.notna(call_data.get('notes')) and call_data.get('notes') != '':
                st.markdown(f"**Notes:** {call_data['notes']}")
