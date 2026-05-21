# Meridian Finance - Dataset Reference

This document details the raw, processed, and sample datasets for both the ML (Bank Marketing) and RAG (CFPB Complaints) lanes.

## Git & Storage Policy

- **Ignored**: Raw and processed datasets are excluded from Git to keep the repository lightweight.
- **Tracked**: 5-row mock samples (`sample_bank_marketing.csv` and `sample_cfpb_complaints.csv`) are tracked to allow immediate test verification and CI/CD validation.

---

## Datasets

### 1. UCI Bank Marketing
* **Task**: Predict term-deposit conversion (`y`) using customer demographics and macroeconomic indicators.
* **Source URL**: [UCI Bank Marketing (bank-additional.zip)](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip)
* **Shape**: 41,188 rows × 21 columns.
* **Key Columns**:
  - `age` (int): Customer age
  - `job` (categorical): Occupation type
  - `y` (categorical): Subscription result (`yes` / `no`) - target variable

### 2. CFPB Consumer Complaints
* **Task**: Retrieve and reason over consumer complaints within the RAG pipeline.
* **Source API**: [CFPB Search API](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/) (filtered with `has_narrative=true`)
* **Shape**: 10,000 unique records (localized sample).
* **Key Columns**:
  - `complaint_id` (int): Unique identifier
  - `consumer_complaint_narrative` (str): Raw text detailing the issue (standardized from `complaint_what_happened` in API response)
  - `product` / `issue` (str): Complaint categorization metadata
