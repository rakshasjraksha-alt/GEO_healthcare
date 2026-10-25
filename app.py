"""
================================================================================
app.py - Interactive Streamlit Clinical Prototype for Privacy-Preserving GEO-RAG
================================================================================
Department of Computer Science & Engineering (AI & ML)
Siddaganga Institute of Technology, Tumakuru
Mini-Project (Phase 1/2) - V Semester

Module Overview:
----------------
This is the interactive web interface for the Generative Engine Optimization (GEO)
Healthcare AI framework. It enables clinicians, evaluators, and examiners to:
1. Input patient queries containing sensitive personal identifiers (PII).
2. Inspect the Pre-Retrieval Privacy Firewall in real-time (Before vs. After view).
3. View the 0% PII Leakage validation badge and detection audit log.
4. Compare retrieval using GEO-structured Q&A chunks (BLUF headers) vs. baseline fixed chunks.
5. Inspect interactive Citation Proof Cards displaying NIH Document ID, source link,
   and cosine match confidence scores.
6. Verify safety mechanisms (similarity threshold refusal and emergency warnings).
================================================================================
"""

import sys
import os
import time
from pathlib import Path
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="GEO Healthcare AI - SIT Tumakuru",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import (
    GroundedClinicalRAG,
    ClinicalRAGResult,
    CitationProofCard,
    GEO_COLLECTION_NAME,
    FIXED_COLLECTION_NAME,
    MANDATORY_DISCLAIMER,
    EMERGENCY_SYMPTOM_WARNING
)

# Custom CSS for clinical styling, badges, and proof cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
        color: #92400E;
    }
    .emergency-box {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
        color: #B91C1C;
        font-weight: 600;
    }
    .badge-passed {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1px solid #10B981;
        display: inline-block;
    }
    .badge-failed {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        border: 1px solid #EF4444;
        display: inline-block;
    }
    .proof-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Clinical AI Engine & Vector DB...")
def load_rag_engine():
    """Initializes and caches the Grounded Clinical RAG engine."""
    return GroundedClinicalRAG()


def main():
    rag_engine = load_rag_engine()

    # Title & Academic Attribution
    st.markdown('<div class="main-header">🏥 Generative Engine Optimization (GEO) Healthcare AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header"><b>Department of Computer Science & Engineering (AI & ML)</b> | '
        'Siddaganga Institute of Technology, Tumakuru — Mini-Project Phase 1/2 (V Semester)</div>',
        unsafe_allow_html=True
    )

    # Mandatory Disclaimers
    st.markdown(f'<div class="disclaimer-box">⚠️ {MANDATORY_DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="emergency-box">🚨 {EMERGENCY_SYMPTOM_WARNING}</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SIDEBAR: Clinical Controls & Preset Queries
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Clinical Configuration")

        strategy_choice = st.radio(
            "Knowledge Structuring Strategy:",
            options=["GEO-Structured (BLUF Headers)", "Fixed-Size Baseline (500 chars)"],
            index=0,
            help="Compare our proposed GEO structuring with BLUF headers against naive sliding-window chunking."
        )
        selected_collection = (
            GEO_COLLECTION_NAME if "GEO" in strategy_choice else FIXED_COLLECTION_NAME
        )

        similarity_threshold = st.slider(
            "Evidence Validation Cutoff:",
            min_value=0.20,
            max_value=0.60,
            value=0.35,
            step=0.05,
            help="Minimum cosine similarity required to answer. Below this cutoff, the system refuses to guess."
        )
        rag_engine.similarity_threshold = similarity_threshold

        top_k = st.slider("Top Evidence Passages (k):", min_value=1, max_value=5, value=2)

        st.markdown("---")
        st.subheader("📋 Preset Evaluation Scenarios")
        preset = st.selectbox(
            "Choose a demo scenario:",
            options=[
                "Custom Query (Type Below)",
                "1. Motivating Scenario (Rahul - Blood Sugar 220 mg/dL)",
                "2. Hypertension Symptoms (Priya - Phone + DOB)",
                "3. Clean Clinical Query (No PII Present)",
                "4. Out-of-Domain Query (Automotive - Should Refuse)"
            ]
        )

    # Determine default text based on preset
    if preset == "1. Motivating Scenario (Rahul - Blood Sugar 220 mg/dL)":
        default_query = "My name is Rahul, DOB 12/04/1990, phone 9876543210. Blood sugar is 220 mg/dL, what should I do?"
    elif preset == "2. Hypertension Symptoms (Priya - Phone + DOB)":
        default_query = "Patient Priya Sharma, DOB: 15/08/1992, mobile +91 9845012345. What are the warning signs of hypertensive crisis?"
    elif preset == "3. Clean Clinical Query (No PII Present)":
        default_query = "What is an A1C blood test and what do the percentage results indicate?"
    elif preset == "4. Out-of-Domain Query (Automotive - Should Refuse)":
        default_query = "How do I replace a transmission belt on a 2012 sedan car?"
    else:
        default_query = ""

    # --------------------------------------------------------------------------
    # QUERY INPUT AREA
    # --------------------------------------------------------------------------
    st.subheader("💬 Patient Inquiry")
    user_query = st.text_area(
        "Enter medical inquiry (with or without personal identifiers):",
        value=default_query,
        height=100,
        placeholder="Type clinical inquiry here... e.g., My name is John, phone 9876543210. Blood sugar is 220 mg/dL..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit_btn = st.button("🚀 Sanitize & Consult AI", type="primary", use_container_width=True)

    if submit_btn and user_query.strip():
        with st.spinner("Executing Privacy Firewall and Clinical RAG Pipeline..."):
            result: ClinicalRAGResult = rag_engine.run(
                user_query=user_query,
                collection_strategy=selected_collection,
                top_k=top_k
            )

        st.markdown("---")

        # ----------------------------------------------------------------------
        # MODULE 1: PRE-RETRIEVAL PRIVACY FIREWALL AUDIT
        # ----------------------------------------------------------------------
        st.subheader("🛡️ Pre-Retrieval Privacy Firewall (Dual-Guard De-Identification)")

        # Metric Badges
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            if result.is_safe:
                st.markdown('<div class="badge-passed">✅ 0.0% PII LEAKAGE (PASSED)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-failed">❌ PII LEAK DETECTED</div>', unsafe_allow_html=True)
        with metric_col2:
            st.metric("PII Entities Scrubbed", f"{result.pii_count} detected")
        with metric_col3:
            st.metric("Firewall Latency", f"{result.latency_ms:.1f} ms")
        with metric_col4:
            st.metric("Strategy Used", "GEO Structured" if "geo" in result.strategy_used else "Fixed Baseline")

        # Before vs After Redaction View
        st.markdown("#### 🔄 Real-Time De-Identification Audit (Before vs. After)")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Raw Input (Contains PII):**")
            st.info(result.query_raw)
        with comp_col2:
            st.markdown("**Sanitized Payload (Stored / Vector Embedded):**")
            st.success(result.query_sanitized)

        st.caption(f"**Clean Semantic Search Query for Vector Store:** `{result.search_query}`")

        # Detailed PII entities expander
        redaction_audit = rag_engine.firewall.sanitize(user_query)
        if redaction_audit.entities_detected:
            with st.expander("🔎 View Detailed De-Identification Entity Provenance"):
                table_data = []
                for e in redaction_audit.entities_detected:
                    table_data.append({
                        "Detected Text": e.text,
                        "Category": e.category,
                        "Firewall Guard": e.guard.upper(),
                        "Span": f"[{e.start_char}:{e.end_char}]",
                        "Replacement Token": e.replacement_token
                    })
                st.dataframe(table_data, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # MODULE 2: EVIDENCE VALIDATION & GROUNDED ANSWER
        # ----------------------------------------------------------------------
        st.subheader("🩺 Grounded Clinical Guidance")

        if result.evidence_valid:
            st.success(f"✅ Evidence Verification Passed (Top match: {result.max_confidence:.4f} ≥ Cutoff {similarity_threshold:.2f})")
        else:
            st.warning(f"⛔ Insufficient Evidence Gate Active (Top match: {result.max_confidence:.4f} < Cutoff {similarity_threshold:.2f})")

        st.markdown(result.answer)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # MODULE 3: INTERACTIVE CITATION PROOF CARDS
        # ----------------------------------------------------------------------
        st.subheader("📚 Verifiable Citation Proof Cards (Provenance Attribution)")
        st.caption("Every clinical statement is grounded in certified NIH MedQuAD documents with traceable confidence scores:")

        if not result.citations:
            st.info("No citations retrieved.")
        else:
            card_cols = st.columns(len(result.citations))
            for idx, (col, card) in enumerate(zip(card_cols, result.citations), start=1):
                with col:
                    st.markdown(f"### Proof Card #{idx}")
                    st.markdown(f"**NIH Doc ID:** `{card.document_id}`")
                    st.markdown(f"**Source:** {card.source}")
                    st.markdown(f"**Focus Area:** {card.focus}")

                    # Cosine confidence progress bar
                    st.markdown(f"**Match Confidence:** `{card.confidence_score:.4f}`")
                    st.progress(min(1.0, max(0.0, card.confidence_score)))

                    if card.url:
                        st.markdown(f"[🔗 Official Source Reference]({card.url})")

                    with st.expander("📄 Supporting Passage Snippet"):
                        st.write(card.snippet)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>"
        "Generative Engine Optimization (GEO) Healthcare AI | Siddaganga Institute of Technology, Tumakuru<br>"
        "Department of Computer Science & Engineering (AI & ML) — Academic Research Mini-Project"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
