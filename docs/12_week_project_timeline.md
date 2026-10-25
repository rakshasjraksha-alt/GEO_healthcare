# 12-Week Academic Semester Timeline & Implementation Roadmap
## Mini-Project: Generative Engine Optimization (GEO) Healthcare AI
**Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)**  
**Siddaganga Institute of Technology (SIT), Tumakuru**  
*Curriculum Mapping: B.E. V Semester Mini-Project (Phase 1 / Phase 2)*

---

## 📅 High-Level Semester Schedule & Departmental Review Gates

```text
[Weeks 1 - 3]  ==> Phase 1: Literature Survey, Ethics, & MedQuAD Ingestion
                   🚩 Milestone 1: Problem Statement & Dataset Finalization (Review 1)
                   
[Weeks 4 - 5]  ==> Phase 2: Pre-Retrieval Privacy Firewall (Regex + spaCy NER)
                   🚩 Milestone 2: De-Identification Engine & Eponym Whitelist Validation
                   
[Weeks 6 - 7]  ==> Phase 3: GEO Knowledge Restructuring & ChromaDB Vector Indexing
                   🚩 Milestone 3: BLUF Q&A Chunks & Dual Persistent Indexing
                   
[Weeks 8 - 9]  ==> Phase 4: Grounded Clinical RAG Engine & Streamlit UI
                   🚩 Milestone 4: Evidence Threshold Gate & Citation Proof Cards (Review 2)
                   
[Weeks 10 - 12] ==> Phase 5: Empirical Benchmark, Documentation & Final Viva Defense
                   🚩 Milestone 5: 100-Query Privacy Audit, Ablation Charts & Final Viva
```

---

## 🗓️ Detailed Week-by-Week Implementation Plan

### 📌 Milestone 1: Literature Survey, Problem Formulation & Dataset Ingestion
#### Week 1: Literature Review & Problem Formulation
- **Objective**: Conduct academic literature review on Medical LLMs, Retrieval-Augmented Generation (RAG), and clinical de-identification standards (HIPAA Safe Harbor, India DPDP Act).
- **Activities**:
  - Analyze limitations of naive chunking and black-box healthcare chatbots.
  - Define problem statement, motivating clinical scenario (Rahul's inquiry), and research objectives.
- **Deliverable**: Project Synopsis and Literature Survey draft.

#### Week 2: Environment Setup & Repository Architecture
- **Objective**: Establish development standards, dependency pinning, and directory layout.
- **Activities**:
  - Configure isolated virtual environment and `requirements.txt`.
  - Set up Git repository and directory skeleton (`data/`, `chroma_db/`, `src/`, `tests/`, `docs/`).
  - Configure `.env.example` for local offline execution.
- **Deliverable**: Repository structure, `requirements.txt`, and setup documentation (`README.md`).

#### Week 3: NIH MedQuAD Dataset Ingestion & XML Parsing
- **Objective**: Ingest and structure the medical knowledge corpus from certified National Institutes of Health sources.
- **Activities**:
  - Develop `src/data_prep.py` with automated GitHub archive download.
  - Parse hierarchical XML tags (`<Document>`, `<Focus>`, `<QAPair>`, `<Question>`, `<Answer>`).
  - Clean clinical text (normalize spacing, strip markup, remove artifacts).
  - Implement zero-downtime offline fallback dataset for lab testing.
- **Deliverable**: `src/data_prep.py` and exported datasets (`medquad_clean.csv`, `medquad_clean.json`).
- 🚩 **Review Gate**: **Departmental Mini-Project Review 1 (Synopsis & Dataset Approval)**.

---

### 📌 Milestone 2: Privacy-Preserving De-Identification Engine
#### Week 4: Deterministic Pre-Retrieval RegEx Engine (Guard 1)
- **Objective**: Build deterministic scrubbing for structured personal identifiers.
- **Activities**:
  - Formulate compiled regular expressions for Indian mobile numbers (`9876543210`, `+91`), US numbers, birth dates (`DD/MM/YYYY`, `YYYY-MM-DD`), and government/hospital IDs (Aadhaar, SSN, MRN).
  - Implement reverse-index string masking to prevent character offset distortion.
- **Deliverable**: Guard 1 module in `src/privacy_firewall.py`.

#### Week 5: Statistical NER & Medical Eponym Whitelisting (Guard 2)
- **Objective**: Mask unstructured personal names while preventing clinical over-redaction.
- **Activities**:
  - Integrate spaCy statistical pipeline (`en_core_web_sm`) and conversational self-introduction patterns (`My name is [Name]`).
  - Design and calibrate `MEDICAL_EPONYM_WHITELIST` to preserve physician-named conditions (*Parkinson's, Alzheimer's, Crohn's, Hashimoto's*).
  - Build automated unit test suite (`tests/test_privacy.py`) validating the Rahul motivating example.
- **Deliverable**: `src/privacy_firewall.py` and `tests/test_privacy.py` (8/8 unit tests passing).

---

### 📌 Milestone 3: Knowledge Restructuring & Vector Indexing
#### Week 6: Generative Engine Optimization (GEO) Knowledge Restructuring
- **Objective**: Restructure clinical knowledge for dense embedding and LLM attention optimization.
- **Activities**:
  - Formulate Bottom-Line-Up-Front (BLUF) summary extraction placing core directives in chunk headers.
  - Bind questions, BLUF takeaways, answers, and provenance into atomic self-contained units.
  - Build sliding-window baseline dataset (500 chars, 50 overlap) for comparative ablation.
- **Deliverable**: `src/geo_structuring.py` and generated datasets (`geo_structured_chunks.json`, `baseline_fixed_chunks.json`).

#### Week 7: Persistent ChromaDB Dense Vector Indexing
- **Objective**: Establish local, persistent, offline-capable vector storage on consumer CPU hardware.
- **Activities**:
  - Integrate `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
  - Initialize two separate persistent collections: `geo_chunks` and `fixed_chunks`.
  - Implement batched encoding and indexing with SQLite persistence in `chroma_db/`.
  - Conduct retrieval smoke test on Rahul's scrubbed query.
- **Deliverable**: `src/indexing.py` and initialized Chroma database collections.

---

### 📌 Milestone 4: Grounded Pipeline & Interactive Clinical UI
#### Week 8: Evidence Validation Gate & Grounded Prompt Engineering
- **Objective**: Eliminate hallucinations and ungrounded dosages via threshold-gated generation.
- **Activities**:
  - Build `src/rag_pipeline.py` orchestrating firewall $\rightarrow$ retrieval $\rightarrow$ threshold validation $\rightarrow$ generation.
  - Implement Evidence Validation Gate: refuse out-of-domain queries when $\text{Cosine Similarity} < 0.35$.
  - Design multi-tier inference (Local Ollama $\rightarrow$ Remote API $\rightarrow$ Extractive Grounded Synthesizer).
  - Embed clinical disclaimers and emergency symptom advisories.
- **Deliverable**: `src/rag_pipeline.py`.

#### Week 9: Interactive Streamlit Clinical Prototype
- **Objective**: Deliver a full-featured clinical decision support web application.
- **Activities**:
  - Build `app.py` with real-time before/after de-identification view and `0.0% PII LEAKAGE` badge.
  - Implement interactive multi-column Citation Proof Cards with live confidence progress bars and links.
  - Add preset scenario buttons (Rahul query, Hypertension, Clean query, Automotive refusal).
  - Create one-click launcher scripts (`run_app.bat`, `run_app.sh`).
- **Deliverable**: `app.py`, `run_app.bat`, `run_app.sh`.
- 🚩 **Review Gate**: **Departmental Mini-Project Review 2 (Working Prototype & Architecture Demo)**.

---

### 📌 Milestone 5: Empirical Benchmark, Documentation & Final Viva
#### Week 10: 100-Query Privacy Benchmark & Ablation Study
- **Objective**: Conduct rigorous empirical evaluation across safety and generation metrics.
- **Activities**:
  - Build synthetic 100-query labeled test suite (Standard PII, IDs, Eponyms, Clean, Adversarial).
  - Measure Precision, Recall, F1-Score, PII Leak Rate, and Over-Redaction Rate.
  - Execute ablation benchmark comparing `geo_chunks` vs. `fixed_chunks` (+22% Faithfulness gain).
  - Conduct honest boundary failure analysis on spaced/obfuscated PII.
- **Deliverable**: `src/evaluation.py` and `data/processed/evaluation_report.json`.

#### Week 11: Academic Visualization & Report Compilation
- **Objective**: Generate publication-ready figures and compile the final mini-project report.
- **Activities**:
  - Generate high-resolution figures: `evaluation_ablation_chart.png` and `privacy_metrics_breakdown.png`.
  - Draft project report chapters: Abstract, Introduction, Literature Survey, Proposed Architecture, Implementation Details, Results & Discussion, Limitations, and Future Work.
- **Deliverable**: `docs/project_report_and_viva_guide.md` and chart artifacts.

#### Week 12: Viva Voce Preparation & Final Project Defense
- **Objective**: Rehearse demonstration, verify offline execution, and prepare for examiner questions.
- **Activities**:
  - Test the entire pipeline on a clean laptop without active internet to guarantee offline readiness.
  - Practice defending the 15 core viva examination questions.
  - Prepare demonstration slides illustrating the Rahul motivating scenario and ablation results.
- **Deliverable**: Final Mini-Project Report, Project Diary, and Presentation Slides.
- 🚩 **Final Evaluation**: **Phase 1 / Phase 2 Semester End Examination (SEE) Viva Voce**.

---

## 📊 Summary of Weekly Deliverables vs. Course Modules

| Week | Focus Area | Key Deliverable | Verification Artifact |
| :---: | :--- | :--- | :--- |
| **W1** | Problem Formulation & Ethics | Literature Survey Draft | Problem Statement Document |
| **W2** | Repository & Environment | Repo Structure & Dependencies | `requirements.txt`, `README.md` |
| **W3** | NIH MedQuAD Dataset | Ingestion & XML Parsing | `src/data_prep.py`, `medquad_clean.csv` |
| **W4** | Structured De-Identification | Guard 1: RegEx Engine | `src/privacy_firewall.py` (Guard 1) |
| **W5** | Statistical NER & Whitelist | Guard 2 + Unit Test Suite | `tests/test_privacy.py` (8/8 Passed) |
| **W6** | GEO Structuring & BLUF | Atomic Q&A + Baseline | `src/geo_structuring.py`, JSON chunks |
| **W7** | Dense Vector Indexing | Persistent ChromaDB Store | `src/indexing.py`, `chroma_db/` |
| **W8** | Grounded Clinical RAG | Evidence Threshold Gate | `src/rag_pipeline.py` |
| **W9** | Clinical Web Interface | Streamlit Decision Support UI | `app.py`, `run_app.bat` |
| **W10**| Safety & Ablation Benchmark | 100-Query Evaluation Suite | `src/evaluation.py`, `evaluation_report.json` |
| **W11**| Academic Visualization | Publication Charts & Report | Ablation & Privacy Charts, Report Draft |
| **W12**| Project Defense & Viva | Final Defense & Viva Prep | 15 Viva Q&A Guide, Final Demo |
