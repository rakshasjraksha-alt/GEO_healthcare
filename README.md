# Generative Engine Optimization (GEO) Framework for Privacy-Preserving Healthcare AI

**Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)**  
**Siddaganga Institute of Technology (SIT), Tumakuru**  
*Mini Project (Phase 1 / Phase 2) — V Semester B.E.*

---

## 📌 Executive Summary

Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) in healthcare suffer from four critical vulnerabilities:
1. **PII Leakage**: Transmitting patient names, dates of birth, phone numbers, and clinical identifiers directly to embedding stores or LLMs.
2. **Clinical Text Fragmentation**: Standard chunking splits sentences and diagnostic tables arbitrarily, degrading retrieval accuracy.
3. **Hallucination & Overconfidence**: Generating ungrounded dosages or diagnoses when evidence is ambiguous.
4. **Opaque Attribution**: Giving medical answers without traceable, verifiable document-level provenance.

This project delivers a **privacy-first, offline-capable Generative Engine Optimization (GEO) framework** tailored for healthcare:
- **Pre-Retrieval Privacy Firewall**: A dual-guard scrubber (deterministic RegEx + statistical spaCy NER) sanitizing PII *before* vector search or embedding generation.
- **GEO Semantic Restructuring**: Transforming raw NIH MedQuAD clinical records into atomic, self-contained Q&A units with Bottom-Line-Up-Front (BLUF) clinical headers and metadata.
- **Strict Evidence-Grounded Generation**: Enforcing minimum cosine similarity thresholds and prompts prohibiting extrapolation.
- **Full Attribution & Auditability**: Interactive citation proof cards displaying NIH document ID, focus area, passage match confidence, and a real-time 0% PII leak validation badge.

---

## 📂 Repository Structure

```text
geo_healthcare_ai/
│
├── data/
│   ├── raw/                 # Downloaded raw MedQuAD XML files from NIH
│   └── processed/           # Parsed tabular dataset & comparative chunking datasets
│
├── chroma_db/               # Persistent local vector database (No cloud storage)
│   ├── geo_chunks/          # Vector collection for GEO + BLUF structured chunks
│   └── fixed_chunks/        # Vector collection for baseline fixed-size chunks (ablation)
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── data_prep.py         # Deliverable 2: MedQuAD XML parsing & dataset extraction
│   ├── privacy_firewall.py  # Deliverable 3: Dual-guard Regex + spaCy NER scrubber
│   ├── geo_structuring.py   # Deliverable 4: BLUF Q&A chunking vs fixed-size baseline
│   ├── indexing.py          # Deliverable 5: Embeddings & dual ChromaDB collections
│   ├── rag_pipeline.py      # Deliverable 6: Retrieval, evidence validation & grounded LLM
│   └── evaluation.py        # Deliverable 8: Privacy benchmark, Ragas & ablation charts
│
├── tests/
│   ├── __init__.py          # Test suite initializer
│   └── test_privacy.py      # Unit tests for PII firewall & Rahul motivating example
│
├── app.py                   # Deliverable 7: Interactive Streamlit clinical interface
├── requirements.txt         # Project dependencies with pinned versions
├── .env.example             # Environment variable template for local/API configuration
└── README.md                # Project documentation, quickstart, and viva prep
```

---

## 🛠️ Step-by-Step Setup Guide

### 1. Prerequisites & Target Hardware
- **Operating System**: Windows 10/11 or Ubuntu Linux 20.04+
- **CPU**: Intel Core i5 / AMD Ryzen 5 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk**: ~20 GB free SSD storage
- **GPU**: Optional (all embeddings and quantized 4-bit/7B models run efficiently on CPU)
- **Python**: 3.10 to 3.12 recommended

---

### 2. Environment Setup

#### On Windows (PowerShell):
```powershell
# 1. Navigate to the project directory
cd C:\Users\Administrator\.gemini\antigravity\scratch\geo_healthcare_ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# (If script execution is disabled, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Download the spaCy English language model
python -m spacy download en_core_web_sm

# 6. Copy the environment configuration template
copy .env.example .env
```

#### On Linux / macOS (Bash):
```bash
# 1. Navigate to the project directory
cd geo_healthcare_ai

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download the spaCy English language model
python3 -m spacy download en_core_web_sm

# 6. Copy the environment configuration template
cp .env.example .env
```

---

### 3. Local LLM Setup (Offline via Ollama)

To run 100% offline and cost-free:
1. Download and install **Ollama** from [https://ollama.com](https://ollama.com).
2. Open a terminal and pull the quantized instruction-tuned model:
   ```bash
   ollama pull mistral:7b-instruct-q4_K_M
   ```
   *(Or for lightweight 8GB RAM systems: `ollama pull phi3:mini` or `ollama pull qwen2:1.5b`)*
3. Ollama runs automatically on `http://localhost:11434`.

*(Note: The framework also provides an automatic fallback to standard OpenAI/Groq/Gemini endpoints if local Ollama is not active).*
