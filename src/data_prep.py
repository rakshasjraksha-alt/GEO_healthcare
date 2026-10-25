"""
================================================================================
data_prep.py - MedQuAD Dataset Ingestion, Parsing, and Cleaning
================================================================================
Department of Computer Science & Engineering (AI & ML)
Siddaganga Institute of Technology, Tumakuru
Mini-Project (Phase 1/2) - V Semester

Module Overview:
----------------
This module handles the ingestion and preprocessing of the National Institutes
of Health (NIH) MedQuAD (Medical Question Answering Dataset). MedQuAD comprises
47,457 medical Q&A pairs extracted from 12 trusted NIH websites (including
NIDDK, CDC, NHLBI, Cancer.gov, and MedlinePlus).

Key Responsibilities:
1. Automated retrieval of MedQuAD XML files from GitHub.
2. Robust XML parsing to extract:
   - document_id : Unique identifier of the clinical source document
   - source      : Publishing institute (e.g., NIDDK, CDC, NHLBI)
   - url         : Official verified source URL
   - focus       : Clinical condition/topic (e.g., "Diabetes", "Hypertension")
   - question_id : Specific question identifier
   - question_type: Category (e.g., "treatment", "symptoms", "causes")
   - question    : Patient/clinical inquiry string
   - answer      : Full evidence-based response from NIH physicians
3. Text cleaning: normalization of whitespaces, removal of markup artifacts.
4. Exporting a structured tabular dataset to CSV and JSON formats.
5. Offline fallback mechanism ensuring the pipeline runs even without an
   active internet connection.
================================================================================
"""

import os
import sys
import re
import io
import json
import logging
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Configure structured logging for academic auditing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataPrep")

# Primary GitHub repository URL for NIH MedQuAD
MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"

# Default high-yield clinical subsets (NIDDK, MedlinePlus, NHLBI, CDC)
# Focusing on these provides high relevance for metabolic, cardiovascular, and primary care
PRIORITY_SUBSETS = [
    "5_NIDDK_QA",                 # Diabetes, Digestive, and Kidney Diseases
    "4_MPlus_Health_Topics_QA",   # MedlinePlus Primary Care & Symptoms
    "8_NHLBI_QA",                 # Heart, Lung, and Blood Institute
    "9_CDC_QA"                    # Centers for Disease Control and Prevention
]


def clean_clinical_text(text: Optional[str]) -> str:
    """
    Sanitizes raw text extracted from MedQuAD XML nodes.
    - Strips leading and trailing whitespaces.
    - Normalizes non-breaking spaces and Unicode whitespace characters.
    - Removes orphan HTML tags if present in the raw text.
    - Collapses consecutive whitespace into a single space.
    """
    if not text:
        return ""
    # Normalize non-breaking spaces and tabs
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Remove any leftover HTML tags (e.g., <p>, <br>, <b>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple spaces or multiple blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def download_medquad(
    target_raw_dir: Path,
    subsets: Optional[List[str]] = None,
    force_download: bool = False
) -> Path:
    """
    Downloads the MedQuAD repository archive from GitHub and extracts XML files.

    Args:
        target_raw_dir: Directory where unzipped XML files will reside.
        subsets: List of folder names to extract (None extracts all).
        force_download: If True, re-downloads even if target exists.

    Returns:
        Path to the root folder containing the extracted XML folders.
    """
    target_raw_dir = Path(target_raw_dir)
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if files are already extracted
    existing_xmls = list(target_raw_dir.glob("**/*.xml"))
    if existing_xmls and not force_download:
        logger.info(f"Existing raw XML files found ({len(existing_xmls)} files). Skipping download.")
        return target_raw_dir

    logger.info(f"Downloading MedQuAD master archive from {MEDQUAD_ZIP_URL}...")
    try:
        req = urllib.request.Request(
            MEDQUAD_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SIT-MiniProject/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            zip_bytes = io.BytesIO(response.read())

        logger.info("Download completed successfully. Extracting XML archives...")
        with zipfile.ZipFile(zip_bytes) as z:
            namelist = z.namelist()
            extracted_count = 0
            for name in namelist:
                if not name.endswith(".xml"):
                    continue

                # Filter by priority subsets if specified
                if subsets:
                    if not any(subset in name for subset in subsets):
                        continue

                # Strip top-level directory (e.g. 'MedQuAD-master/')
                parts = Path(name).parts
                if len(parts) > 1:
                    relative_path = Path(*parts[1:])
                else:
                    relative_path = Path(name)

                dest_file = target_raw_dir / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_file, "wb") as f_out:
                    f_out.write(z.read(name))
                extracted_count += 1

        logger.info(f"Successfully extracted {extracted_count} XML documents into {target_raw_dir}")
        return target_raw_dir

    except Exception as e:
        logger.warning(f"Could not download MedQuAD from GitHub: {e}")
        logger.info("Initializing offline fallback dataset with representative clinical records...")
        _create_offline_fallback(target_raw_dir)
        return target_raw_dir


def _create_offline_fallback(target_raw_dir: Path) -> None:
    """
    Creates high-quality representative MedQuAD XML files for offline demonstration,
    testing, or college lab environments where GitHub is unreachable.
    Focuses specifically on metabolic health (Blood Sugar, Diabetes), cardiovascular care,
    and hypertension to support the motivating scenario and evaluation.
    """
    fallback_dir = target_raw_dir / "5_NIDDK_QA"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    sample_xml_data = [
        {
            "id": "NIDDK-0001",
            "source": "NIDDK",
            "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes/blood-glucose-levels",
            "focus": "Blood Glucose Management and Hyperglycemia",
            "qas": [
                {
                    "qid": "NIDDK-0001-1",
                    "qtype": "treatment",
                    "question": "What should I do if my blood sugar is 220 mg/dL?",
                    "answer": (
                        "A blood glucose reading of 220 mg/dL indicates hyperglycemia (high blood sugar). "
                        "Typical target blood sugar before a meal is 80 to 130 mg/dL, and less than 180 mg/dL "
                        "two hours after starting a meal. If your blood sugar is 220 mg/dL, take the following "
                        "immediate steps: (1) Drink plenty of plain water to help flush excess glucose through urine "
                        "and avoid dehydration; (2) Check your urine for ketones if recommended by your physician, "
                        "especially in type 1 diabetes; (3) Follow your healthcare provider's prescribed insulin or "
                        "medication correction protocol; (4) Avoid strenuous exercise if ketones are present; "
                        "(5) Re-test blood glucose in 1 to 2 hours. Seek immediate emergency medical care if you "
                        "experience shortness of breath, nausea, vomiting, confusion, or fruity breath odor."
                    )
                },
                {
                    "qid": "NIDDK-0001-2",
                    "qtype": "symptoms",
                    "question": "What are the warning signs of dangerously high blood sugar?",
                    "answer": (
                        "Common symptoms of high blood sugar (hyperglycemia) include increased thirst, frequent "
                        "urination, blurred vision, fatigue, and headache. Severe hyperglycemia can progress to "
                        "Diabetic Ketoacidosis (DKA) or Hyperosmolar Hyperglycemic State (HHS). Warning signs "
                        "requiring immediate emergency medical attention include rapid breathing, nausea, vomiting, "
                        "extreme confusion, extreme lethargy, abdominal pain, and a fruity acetone odor to the breath."
                    )
                },
                {
                    "qid": "NIDDK-0001-3",
                    "qtype": "information",
                    "question": "What are normal target blood sugar ranges for adults with diabetes?",
                    "answer": (
                        "According to the American Diabetes Association and NIH guidelines, target blood glucose "
                        "levels for non-pregnant adults with diabetes are generally: Fasting or pre-meal blood "
                        "glucose between 80 and 130 mg/dL, and post-prandial (1 to 2 hours after the start of a meal) "
                        "blood glucose under 180 mg/dL. Individual targets may be customized by a physician based on "
                        "age, duration of diabetes, and comorbid medical conditions."
                    )
                }
            ]
        },
        {
            "id": "NHLBI-0002",
            "source": "NHLBI",
            "url": "https://www.nhlbi.nih.gov/health-topics/high-blood-pressure",
            "focus": "Hypertension (High Blood Pressure)",
            "qas": [
                {
                    "qid": "NHLBI-0002-1",
                    "qtype": "treatment",
                    "question": "How is high blood pressure treated and managed?",
                    "answer": (
                        "Treatment for high blood pressure combines lifestyle modifications and prescription medications. "
                        "Key lifestyle changes include following the DASH (Dietary Approaches to Stop Hypertension) diet "
                        "rich in fruits, vegetables, and low-fat dairy, reducing daily sodium intake to less than 2,300 mg "
                        "(ideally 1,500 mg), engaging in at least 150 minutes of moderate aerobic activity weekly, "
                        "maintaining a healthy body weight, limiting alcohol consumption, and quitting smoking. "
                        "When medications are needed, doctors often prescribe ACE inhibitors, ARBs, calcium channel blockers, "
                        "or thiazide diuretics."
                    )
                },
                {
                    "qid": "NHLBI-0002-2",
                    "qtype": "symptoms",
                    "question": "What are the symptoms of a hypertensive emergency crisis?",
                    "answer": (
                        "High blood pressure is often a 'silent killer' with no symptoms until organ damage occurs. "
                        "However, a hypertensive crisis (systolic pressure above 180 mmHg and/or diastolic pressure above 120 mmHg) "
                        "can cause severe headache, shortness of breath, nosebleeds, chest pain, back pain, numbness or weakness, "
                        "and difficulty speaking. This is an acute medical emergency requiring immediate hospitalization."
                    )
                }
            ]
        },
        {
            "id": "CDC-0003",
            "source": "CDC",
            "url": "https://www.cdc.gov/diabetes/basics/type2.html",
            "focus": "Type 2 Diabetes Prevention and Lifestyle",
            "qas": [
                {
                    "qid": "CDC-0003-1",
                    "qtype": "causes",
                    "question": "What causes Type 2 Diabetes and what are the primary risk factors?",
                    "answer": (
                        "Type 2 diabetes occurs when body cells become resistant to the action of insulin, and the pancreas "
                        "is unable to produce sufficient insulin to overcome this resistance. Risk factors include being overweight "
                        "or obese, being age 45 or older, having a family history of type 2 diabetes, physical inactivity, "
                        "a history of gestational diabetes, and belonging to certain racial or ethnic groups (African American, "
                        "Hispanic/Latino, American Indian, or Asian American)."
                    )
                }
            ]
        }
    ]

    for doc in sample_xml_data:
        root = ET.Element(
            "Document",
            id=doc["id"],
            source=doc["source"],
            url=doc["url"]
        )
        focus_elem = ET.SubElement(root, "Focus")
        focus_elem.text = doc["focus"]

        qa_pairs_elem = ET.SubElement(root, "QAPairs")
        for idx, qa in enumerate(doc["qas"], start=1):
            pair_elem = ET.SubElement(qa_pairs_elem, "QAPair", pid=str(idx))
            q_elem = ET.SubElement(
                pair_elem,
                "Question",
                qid=qa["qid"],
                qtype=qa["qtype"]
            )
            q_elem.text = qa["question"]
            a_elem = ET.SubElement(pair_elem, "Answer")
            a_elem.text = qa["answer"]

        tree = ET.ElementTree(root)
        file_path = fallback_dir / f"{doc['id']}.xml"
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    logger.info(f"Created fallback sample XMLs in {fallback_dir}")


def parse_medquad_xml(xml_file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a single MedQuAD XML file and extracts all contained Q&A pairs
    along with document-level and pair-level clinical metadata.

    Args:
        xml_file_path: Path to the target XML file.

    Returns:
        List of dictionaries, each representing a single Q&A record.
    """
    records: List[Dict[str, Any]] = []

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Extract Document-level attributes
        doc_id = root.attrib.get("id", xml_file_path.stem)
        source = root.attrib.get("source", "NIH")
        url = root.attrib.get("url", "")

        # Extract Focus topic (disease, condition, or drug)
        focus_elem = root.find("Focus")
        focus = clean_clinical_text(focus_elem.text) if focus_elem is not None else ""

        # Find all QAPair elements
        qa_pairs = root.findall(".//QAPair")
        for qa in qa_pairs:
            pid = qa.attrib.get("pid", "")

            # Extract Question element and attributes
            q_elem = qa.find("Question")
            if q_elem is not None:
                qid = q_elem.attrib.get("qid", f"{doc_id}-{pid}")
                qtype = q_elem.attrib.get("qtype", "general")
                question_text = clean_clinical_text(q_elem.text)
            else:
                qid = f"{doc_id}-{pid}"
                qtype = "general"
                question_text = ""

            # Extract Answer element
            a_elem = qa.find("Answer")
            answer_text = clean_clinical_text(a_elem.text) if a_elem is not None else ""

            # Filter out empty or trivially short answers
            if not question_text or not answer_text or len(answer_text) < 15:
                continue

            records.append({
                "document_id": doc_id,
                "source": source,
                "url": url,
                "focus": focus,
                "question_id": qid,
                "question_type": qtype,
                "question": question_text,
                "answer": answer_text,
                "xml_source_file": xml_file_path.name
            })

    except ET.ParseError as pe:
        logger.warning(f"XML Parse error in {xml_file_path.name}: {pe}")
    except Exception as ex:
        logger.error(f"Unexpected error processing {xml_file_path.name}: {ex}")

    return records


def build_dataset(
    raw_dir: Path,
    output_csv_path: Path,
    output_json_path: Path,
    max_records: Optional[int] = None,
    focus_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Parses all XML files in raw_dir, constructs a unified pandas DataFrame,
    and exports it to CSV and JSON formats.

    Args:
        raw_dir: Path containing raw MedQuAD XML files.
        output_csv_path: Target path for the output CSV file.
        output_json_path: Target path for the output JSON file.
        max_records: Maximum number of records to ingest (useful for quick experiments).
        focus_filter: Optional regex or substring to select specific conditions (e.g. 'Diabetes').

    Returns:
        pd.DataFrame containing the parsed dataset.
    """
    raw_dir = Path(raw_dir)
    all_xml_files = list(raw_dir.glob("**/*.xml"))
    logger.info(f"Found {len(all_xml_files)} XML files to process in {raw_dir}.")

    all_records: List[Dict[str, Any]] = []

    for xml_path in all_xml_files:
        records = parse_medquad_xml(xml_path)
        for r in records:
            if focus_filter:
                if not re.search(focus_filter, r["focus"], re.IGNORECASE):
                    continue
            all_records.append(r)
            if max_records and len(all_records) >= max_records:
                break
        if max_records and len(all_records) >= max_records:
            break

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No records were extracted! Creating fallback records...")
        _create_offline_fallback(raw_dir)
        return build_dataset(raw_dir, output_csv_path, output_json_path, max_records, focus_filter)

    # Ensure parent output directories exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to CSV and JSON
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    df.to_json(output_json_path, orient="records", indent=2, force_ascii=False)

    logger.info(f"Dataset successfully exported:")
    logger.info(f"  - CSV : {output_csv_path} ({len(df)} rows)")
    logger.info(f"  - JSON: {output_json_path}")

    return df


def display_dataset_summary(df: pd.DataFrame) -> None:
    """
    Prints a formatted academic summary table of the parsed MedQuAD dataset,
    suitable for mini-project presentation slides and project reports.
    """
    print("\n" + "=" * 70)
    print("MEDQUAD DATASET INGESTION SUMMARY (SIT TUMAKURU - MINI PROJECT)")
    print("=" * 70)
    print(f"Total Medical Q&A Pairs Ingested : {len(df):,}")
    print(f"Total Unique Source Documents    : {df['document_id'].nunique():,}")
    print(f"Total Unique Focus Areas/Topics  : {df['focus'].nunique():,}")
    print(f"Average Answer Length (chars)   : {df['answer'].str.len().mean():.1f}")
    print(f"Average Question Length (chars) : {df['question'].str.len().mean():.1f}")
    print("-" * 70)

    print("Breakdown by NIH Source:")
    source_counts = df["source"].value_counts().head(8)
    for src, count in source_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • {src:<25} : {count:>5} records ({pct:5.1f}%)")

    print("-" * 70)
    print("Top Question Types:")
    qtype_counts = df["question_type"].value_counts().head(8)
    for qt, count in qtype_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • {qt:<25} : {count:>5} records ({pct:5.1f}%)")

    print("-" * 70)
    print("Sample Ingested Record (Motivating Scenario Relevance):")
    sample = df[df["focus"].str.contains("Blood Glucose|Diabetes", case=False, na=False)].head(1)
    if sample.empty:
        sample = df.head(1)

    row = sample.iloc[0]
    print(f"  Doc ID   : {row['document_id']} ({row['source']})")
    print(f"  Focus    : {row['focus']}")
    print(f"  QType    : {row['question_type']}")
    print(f"  Question : {row['question']}")
    print(f"  Answer   : {row['answer'][:200]}...")
    print(f"  URL      : {row['url']}")
    print("=" * 70 + "\n")


def main():
    """
    Main driver execution for Deliverable 2:
    1. Sets up directory paths.
    2. Downloads MedQuAD subsets (prioritizing NIDDK, MedlinePlus, NHLBI, CDC).
    3. Parses XML files and cleans clinical text.
    4. Exports clean tabular dataset.
    5. Displays summary statistics.
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "medquad"
    processed_dir = project_root / "data" / "processed"
    output_csv = processed_dir / "medquad_clean.csv"
    output_json = processed_dir / "medquad_clean.json"

    logger.info("Initiating Deliverable 2: MedQuAD Download and Parse...")

    # Step 1: Download MedQuAD priority clinical subsets
    download_medquad(
        target_raw_dir=raw_dir,
        subsets=PRIORITY_SUBSETS,
        force_download=False
    )

    # Step 2: Build clean tabular dataset (limit to 1500 for fast local indexing, or None for full)
    df = build_dataset(
        raw_dir=raw_dir,
        output_csv_path=output_csv,
        output_json_path=output_json,
        max_records=1500  # Optimal sample size for fast, responsive student laptop embedding
    )

    # Step 3: Display academic summary
    display_dataset_summary(df)


if __name__ == "__main__":
    main()
