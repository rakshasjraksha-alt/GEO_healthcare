"""
================================================================================
test_privacy.py - Unit Tests for Pre-Retrieval Privacy Firewall
================================================================================
Department of Computer Science & Engineering (AI & ML)
Siddaganga Institute of Technology, Tumakuru
Mini-Project (Phase 1/2) - V Semester

Tests:
1. Motivating example verification (Rahul query)
2. Deterministic RegEx Phone scrubbing (Indian + International)
3. Date of Birth scrubbing (various date formats)
4. Identification Numbers scrubbing (Aadhaar, SSN, MRN)
5. Email address scrubbing
6. spaCy / Contextual NER personal name masking
7. Medical Eponym Preservation (Prevents Over-Redaction of Alzheimer's, Parkinson's)
8. Clean medical query preservation (Zero False Positives)
9. Sanitized search query formulation
================================================================================
"""

import sys
from pathlib import Path
import pytest

# Ensure src directory is accessible for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.privacy_firewall import PreRetrievalPrivacyFirewall, RedactionResult


@pytest.fixture(scope="module")
def firewall():
    """Initializes a shared PreRetrievalPrivacyFirewall instance."""
    return PreRetrievalPrivacyFirewall()


def test_motivating_example(firewall):
    """
    Validates the exact motivating scenario from the project specification:
    Input: "My name is Rahul, DOB 12/04/1990, phone 9876543210. Blood sugar is 220 mg/dL, what should I do?"
    Expected: Names, DOB, and phone are scrubbed; clinical condition is preserved.
    """
    raw_query = "My name is Rahul, DOB 12/04/1990, phone 9876543210. Blood sugar is 220 mg/dL, what should I do?"
    result = firewall.sanitize(raw_query)

    # 1. Check raw identifiers are masked
    assert "Rahul" not in result.sanitized_text, "Patient name Rahul was not redacted!"
    assert "12/04/1990" not in result.sanitized_text, "DOB 12/04/1990 was not redacted!"
    assert "9876543210" not in result.sanitized_text, "Phone 9876543210 was not redacted!"

    # 2. Check clinical content is preserved
    assert "220 mg/dL" in result.sanitized_text, "Clinical blood sugar value was accidentally removed!"
    assert "blood sugar" in result.sanitized_text.lower()

    # 3. Check privacy audit status
    assert result.is_safe is True
    assert result.pii_count >= 3
    assert result.leak_rate_pct == 0.0


def test_phone_numbers_redaction(firewall):
    """Tests various phone number patterns: 10-digit Indian, +91 prefixed, US format."""
    queries = [
        ("Call me at 9876543210 regarding insulin", "[REDACTED_PHONE]"),
        ("Phone is +91 9845012345 urgent", "[REDACTED_PHONE]"),
        ("Contact: (555) 234-5678 doctor", "[REDACTED_PHONE]"),
        ("Mobile 91234-56789 clinic", "[REDACTED_PHONE]")
    ]
    for text, expected_token in queries:
        res = firewall.sanitize(text)
        assert expected_token in res.sanitized_text
        assert any(e.category == "PHONE" for e in res.entities_detected)


def test_date_of_birth_redaction(firewall):
    """Tests date of birth patterns with prefix and standalone dates."""
    queries = [
        "DOB: 15/08/1995 need allergy medication",
        "Date of birth 1988-11-23 hypertension check",
        "Born on 12/04/1990 headache symptom"
    ]
    for q in queries:
        res = firewall.sanitize(q)
        assert "[REDACTED_DOB]" in res.sanitized_text
        assert any(e.category == "DOB" for e in res.entities_detected)


def test_id_numbers_redaction(firewall):
    """Tests Government IDs, SSN, and Medical Record Numbers (MRN)."""
    queries = [
        ("My Aadhaar is 5432 1098 7654 for verification", "[REDACTED_ID]"),
        ("Patient MRN-98421 admitted with chest pain", "[REDACTED_ID]"),
        ("SSN number is 123-45-6789", "[REDACTED_ID]")
    ]
    for text, expected_token in queries:
        res = firewall.sanitize(text)
        assert expected_token in res.sanitized_text
        assert any(e.category == "ID_NUMBER" for e in res.entities_detected)


def test_email_redaction(firewall):
    """Tests standard email address de-identification."""
    res = firewall.sanitize("Send lab report to test.patient101@hospital.org please.")
    assert "[REDACTED_EMAIL]" in res.sanitized_text
    assert "test.patient101@hospital.org" not in res.sanitized_text


def test_medical_eponym_preservation_no_over_redaction(firewall):
    """
    CRITICAL MEDICAL TEST:
    Ensures that conditions named after doctors/patients are NOT falsely redacted as PERSON.
    E.g. Alzheimer's disease, Parkinson's disease, Crohn's disease.
    """
    medical_query = "What are the early motor symptoms of Parkinson's disease and Crohn's disease?"
    res = firewall.sanitize(medical_query)

    assert "Parkinson's" in res.sanitized_text or "parkinson" in res.sanitized_text.lower(), \
        "Over-redaction alert: Parkinson's was mistakenly redacted as a person!"
    assert "Crohn's" in res.sanitized_text or "crohn" in res.sanitized_text.lower(), \
        "Over-redaction alert: Crohn's was mistakenly redacted as a person!"


def test_clean_medical_query_zero_false_positives(firewall):
    """Tests that a clean clinical query with no PII remains 100% untouched."""
    clean_query = "What is the recommended fasting blood sugar level for healthy adults?"
    res = firewall.sanitize(clean_query)

    assert res.pii_count == 0
    assert res.sanitized_text.strip() == clean_query.strip()
    assert res.is_safe is True


def test_search_query_optimization(firewall):
    """Tests that sanitized text is synthesized into a concise semantic search query."""
    raw = "My name is Priya, DOB 01/01/1992. What are the symptoms of high cholesterol?"
    res = firewall.sanitize(raw)

    assert "[REDACTED_USER]" in res.clean_search_query
    assert "symptoms of high cholesterol" in res.clean_search_query.lower()
