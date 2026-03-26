import streamlit as st
import pandas as pd
from pathlib import Path


def render_call_card(call_type, call_rows, paper_spects_dir, 
                    show_details=True, show_online=False, score=None, matched=None, card_idx=None):
    """
    Render a call card with spectrogram and metadata.
    
    Args:
        call_type: Call type string (e.g. 'N01i')
        call_rows: DataFrame of all catalogue rows for this call type
        paper_spects_dir: Path to paper spectrograms
        show_details: Whether to show part descriptions
        show_online: Whether to show online examples
        score: Optional score to display (for search results)
        matched: Optional matched tokens string to display
        card_idx: Optional unique index for this card (for stable button keys)
    """
    # Generate stable key suffix
    if card_idx is not None:
        key_suffix = f"{call_type}_{card_idx}"
    else:
        key_suffix = call_type
    
    # Initialize state for online carousel
    if 'current_online_idx' not in st.session_state or not isinstance(st.session_state.current_online_idx, dict):
        st.session_state.current_online_idx = {}
    if call_type not in st.session_state.current_online_idx:
        st.session_state.current_online_idx[call_type] = 0
    
    with st.container(border=True):
        # Display call name as clickable link
        if st.button(f"**{call_type}**", key=f"nav_{key_suffix}", use_container_width=True):
            st.query_params['call_name'] = call_type
            st.switch_page("pages/1_Label_Calls.py")
        
        # Show score and matched tokens if provided (for search results)
        if score is not None:
            st.caption(f"Score: {score:.2f}")
        if matched:
            st.caption(f"Matched: {matched}")
        
        # Show metadata from first row
        first_row = call_rows.iloc[0]
        clans = first_row.get('clan', '')
        pods = first_row.get('pod', '')
        st.caption(f"Clan: {clans} | Pod: {pods}")
        
        # Display paper spectrogram
        paper_spect = paper_spects_dir / f"{call_type}_paper_spect.png"
        if paper_spect.exists():
            st.image(str(paper_spect), use_container_width=True)
        else:
            st.warning(f"Paper spect not found")
        
        # Show online examples if toggled
        if show_online and len(call_rows) > 0:
            current_online_idx = st.session_state.current_online_idx[call_type]
            
            # Ensure index is within bounds
            if current_online_idx >= len(call_rows):
                current_online_idx = 0
                st.session_state.current_online_idx[call_type] = 0
            
            # Get current online row
            current_row = call_rows.iloc[current_online_idx]
            spect_path = Path(current_row['spect_fp'])
            audio_path = current_row.get('audio_fp', '')
            
            st.markdown("---")
            
            # Display online spectrogram
            if spect_path.exists():
                clan_info = current_row.get('clan', 'N/A')
                pod_info = current_row.get('pod', 'N/A')
                annotated = current_row.get('Annotated', False)
                annotated_badge = " ✅" if annotated else ""
                st.image(str(spect_path), use_container_width=True, 
                        caption=f"Example {current_online_idx+1} / {len(call_rows)} | Clan: {clan_info} | Pod: {pod_info}{annotated_badge}")
                
                # Audio player
                if audio_path and Path(audio_path).exists():
                    st.audio(str(audio_path))
                
                # Navigation
                if len(call_rows) > 1:
                    nav_cols = st.columns([1, 1])
                    with nav_cols[0]:
                        if st.button("◀", key=f"prev_{key_suffix}", use_container_width=True):
                            st.session_state.current_online_idx[call_type] = (current_online_idx - 1) % len(call_rows)
                            st.rerun()
                    with nav_cols[1]:
                        if st.button("▶", key=f"next_{key_suffix}", use_container_width=True):
                            st.session_state.current_online_idx[call_type] = (current_online_idx + 1) % len(call_rows)
                            st.rerun()
            else:
                st.warning(f"Online spect not found: {spect_path}")
            
            # Show part descriptions for the current example
            if show_details:
                _render_part_descriptions(current_row)
        
        elif show_details:
            # No online examples shown, show parts from first row
            _render_part_descriptions(first_row)


def _render_part_descriptions(row):
    """Render P1-P5, BIPHO, EXTRA descriptions for a given catalogue row."""
    st.markdown("**Part Labels:**")
    
    part_cols = [f'P{i}' for i in range(1, 6)] + ['BIPHO', 'EXTRA']
    parts_desc = []
    
    for col in part_cols:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            # Include timestamps if available
            start = row.get(f'{col}_start')
            end = row.get(f'{col}_end')
            time_str = ""
            if pd.notna(start) and pd.notna(end):
                time_str = f" `[{float(start):.2f}s - {float(end):.2f}s]`"
            parts_desc.append(f"**{col}:** {val}{time_str}")
    
    if parts_desc:
        st.markdown("<br>".join(parts_desc), unsafe_allow_html=True)
    else:
        st.caption("No part labels for this example")
    
    # Display notes
    note = row.get('note')
    if pd.notna(note) and str(note).strip():
        st.markdown(f"**Note:** {note}")
