import pytest
import pandas as pd
import pandera.errors as pe
from src.data_pipeline.validate import bank_marketing_schema, cfpb_complaints_schema


# Tests for Bank Marketing Schema

@pytest.fixture
def valid_bank_row():
    return {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 150,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.25,
        "nr.employed": 5099.1,
        "y": "no"
    }

def test_bank_marketing_valid(valid_bank_row):
    df = pd.DataFrame([valid_bank_row])
    # Should validate without error
    validated = bank_marketing_schema.validate(df)
    assert len(validated) == 1

def test_bank_marketing_invalid_age(valid_bank_row):
    valid_bank_row["age"] = -5  # Invalid age
    df = pd.DataFrame([valid_bank_row])
    with pytest.raises(pe.SchemaError):
        bank_marketing_schema.validate(df)

def test_bank_marketing_invalid_job(valid_bank_row):
    valid_bank_row["job"] = "astronaut"  # Job not in list
    df = pd.DataFrame([valid_bank_row])
    with pytest.raises(pe.SchemaError):
        bank_marketing_schema.validate(df)

def test_bank_marketing_invalid_target(valid_bank_row):
    valid_bank_row["y"] = "maybe"  # Invalid y value
    df = pd.DataFrame([valid_bank_row])
    with pytest.raises(pe.SchemaError):
        bank_marketing_schema.validate(df)

# Tests for CFPB Complaints Schema

@pytest.fixture
def valid_cfpb_row():
    return {
        "complaint_id": 1234567,
        "date_received": "2023-01-01",
        "product": "Credit card",
        "sub_product": "General-purpose credit card",
        "issue": "Billing dispute",
        "sub_issue": "Incorrect billing amount",
        "consumer_complaint_narrative": "They charged me twice for the same transaction.",
        "company": "Test Bank Inc.",
        "state": "CA",
        "zip_code": "90210",
        "submitted_via": "Web",
        "date_sent_to_company": "2023-01-02",
        "company_response": "Closed with explanation",
        "timely": "Yes",
        "consumer_disputed": "No"
    }

def test_cfpb_complaints_valid(valid_cfpb_row):
    df = pd.DataFrame([valid_cfpb_row])
    validated = cfpb_complaints_schema.validate(df)
    assert len(validated) == 1

def test_cfpb_complaints_missing_narrative(valid_cfpb_row):
    valid_cfpb_row["consumer_complaint_narrative"] = None  # None not allowed
    df = pd.DataFrame([valid_cfpb_row])
    with pytest.raises(pe.SchemaError):
        cfpb_complaints_schema.validate(df)

def test_cfpb_complaints_empty_narrative(valid_cfpb_row):
    valid_cfpb_row["consumer_complaint_narrative"] = "   "  # Empty string not allowed
    df = pd.DataFrame([valid_cfpb_row])
    with pytest.raises(pe.SchemaError):
        cfpb_complaints_schema.validate(df)

def test_cfpb_complaints_invalid_timely(valid_cfpb_row):
    valid_cfpb_row["timely"] = "Sometimes"  # Invalid timely value
    df = pd.DataFrame([valid_cfpb_row])
    with pytest.raises(pe.SchemaError):
        cfpb_complaints_schema.validate(df)
