import os
import argparse
import pandas as pd
import pandera.pandas as pa

# Define Bank Marketing Schema
bank_marketing_schema = pa.DataFrameSchema({
    "age": pa.Column(int, checks=pa.Check.ge(0)),
    "job": pa.Column(str, checks=pa.Check.isin([
        'housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
        'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'
    ])),
    "marital": pa.Column(str, checks=pa.Check.isin(['married', 'single', 'divorced', 'unknown'])),
    "education": pa.Column(str, checks=pa.Check.isin([
        'basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course',
        'unknown', 'university.degree', 'illiterate'
    ])),
    "default": pa.Column(str, checks=pa.Check.isin(['no', 'unknown', 'yes'])),
    "housing": pa.Column(str, checks=pa.Check.isin(['no', 'yes', 'unknown'])),
    "loan": pa.Column(str, checks=pa.Check.isin(['no', 'yes', 'unknown'])),
    "contact": pa.Column(str, checks=pa.Check.isin(['telephone', 'cellular'])),
    "month": pa.Column(str, checks=pa.Check.isin([
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])),
    "day_of_week": pa.Column(str, checks=pa.Check.isin(['mon', 'tue', 'wed', 'thu', 'fri'])),
    "duration": pa.Column(int, checks=pa.Check.ge(0)),
    "campaign": pa.Column(int, checks=pa.Check.ge(1)),
    "pdays": pa.Column(int, checks=pa.Check.ge(0)),
    "previous": pa.Column(int, checks=pa.Check.ge(0)),
    "poutcome": pa.Column(str, checks=pa.Check.isin(['nonexistent', 'failure', 'success'])),
    "emp.var.rate": pa.Column(float),
    "cons.price.idx": pa.Column(float, checks=pa.Check.gt(0)),
    "cons.conf.idx": pa.Column(float),
    "euribor3m": pa.Column(float, checks=pa.Check.gt(0)),
    "nr.employed": pa.Column(float, checks=pa.Check.gt(0)),
    "y": pa.Column(str, checks=pa.Check.isin(['no', 'yes']))
}, coerce=True)

# Define CFPB Complaints Schema
cfpb_complaints_schema = pa.DataFrameSchema({
    "complaint_id": pa.Column(int),
    "date_received": pa.Column(str),
    "product": pa.Column(str),
    "sub_product": pa.Column(str, nullable=True),
    "issue": pa.Column(str),
    "sub_issue": pa.Column(str, nullable=True),
    "consumer_complaint_narrative": pa.Column(str, checks=pa.Check(lambda s: s.str.strip().str.len() > 0)),
    "company": pa.Column(str),
    "state": pa.Column(str, nullable=True),
    "zip_code": pa.Column(str, nullable=True),
    "submitted_via": pa.Column(str),
    "date_sent_to_company": pa.Column(str),
    "company_response": pa.Column(str),
    "timely": pa.Column(str, checks=pa.Check.isin(['Yes', 'No'])),
    "consumer_disputed": pa.Column(str, nullable=True)
}, coerce=True)

def validate_datasets(input_dir: str, output_dir: str):
    """
    Validates Bank Marketing and CFPB Complaints datasets.
    Saves successfully validated datasets to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bank_input_path = os.path.join(input_dir, "bank_marketing.csv")
    bank_output_path = os.path.join(output_dir, "bank_marketing_validated.csv")
    
    cfpb_input_path = os.path.join(input_dir, "cfpb_complaints.csv")
    cfpb_output_path = os.path.join(output_dir, "cfpb_complaints_validated.csv")
    
    # Validate Bank Marketing
    if os.path.exists(bank_input_path):
        print(f"Validating Bank Marketing dataset from {bank_input_path}...")
        df_bank = pd.read_csv(bank_input_path)
        try:
            validated_bank = bank_marketing_schema.validate(df_bank)
            validated_bank.to_csv(bank_output_path, index=False)
            print(f"Successfully validated Bank Marketing and saved to {bank_output_path}.")
        except Exception as e:
            print(f"Bank Marketing validation failed: {e}")
            raise e
    else:
        print(f"Bank Marketing raw file not found at {bank_input_path}. Skipping.")
        
    # Validate CFPB Complaints
    if os.path.exists(cfpb_input_path):
        print(f"Validating CFPB complaints dataset from {cfpb_input_path}...")
        df_cfpb = pd.read_csv(cfpb_input_path)
        try:
            validated_cfpb = cfpb_complaints_schema.validate(df_cfpb)
            validated_cfpb.to_csv(cfpb_output_path, index=False)
            print(f"Successfully validated CFPB complaints and saved to {cfpb_output_path}.")
        except Exception as e:
            print(f"CFPB complaints validation failed: {e}")
            raise e
    else:
        print(f"CFPB complaints raw file not found at {cfpb_input_path}. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ingested datasets.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory with raw data.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save validated data.")
    
    args = parser.parse_args()
    
    validate_datasets(args.input_dir, args.output_dir)
