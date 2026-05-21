import os
import argparse
import urllib.request
import zipfile
import io
import pandas as pd
import requests

UCI_BANK_MARKETING_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
CFPB_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

def download_bank_marketing(output_dir: str):
    """
    Downloads UCI Bank Marketing dataset, extracts 'bank-additional-full.csv',
    and saves it to output_dir/bank_marketing.csv.
    """
    print(f"Downloading Bank Marketing dataset from {UCI_BANK_MARKETING_URL}...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bank_marketing.csv")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(UCI_BANK_MARKETING_URL, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        zip_data = response.read()
        
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
        # Search for bank-additional-full.csv inside the zip
        target_file = None
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith("bank-additional-full.csv"):
                target_file = file_info.filename
                break
        
        if not target_file:
            raise FileNotFoundError("Could not find 'bank-additional-full.csv' inside the zip archive.")
            
        print(f"Extracting {target_file}...")
        with zip_ref.open(target_file) as f_in:
            df = pd.read_csv(f_in, sep=";")
            df.to_csv(output_path, index=False)
            print(f"Saved Bank Marketing dataset to {output_path} ({len(df)} records).")


def fetch_cfpb_complaints(output_dir: str, sample_size: int = 10000):
    """
    Fetches CFPB consumer complaints dataset sample where narrative is present,
    standardizes columns, and saves as output_dir/cfpb_complaints.csv.
    """
    print(f"Fetching {sample_size} CFPB Consumer Complaints with narratives...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cfpb_complaints.csv")
    
    complaints = []
    page_size = 2000
    
    # Define mapping of fields
    fields_mapping = {
        "complaint_id": "complaint_id",
        "date_received": "date_received",
        "product": "product",
        "sub_product": "sub_product",
        "issue": "issue",
        "sub_issue": "sub_issue",
        "complaint_what_happened": "consumer_complaint_narrative",
        "consumer_complaint_narrative": "consumer_complaint_narrative",
        "company": "company",
        "state": "state",
        "zip_code": "zip_code",
        "submitted_via": "submitted_via",
        "date_sent_to_company": "date_sent_to_company",
        "company_response": "company_response",
        "timely": "timely",
        "consumer_disputed": "consumer_disputed"
    }
    
    cursor = None
    
    while len(complaints) < sample_size:
        current_size = min(page_size, sample_size - len(complaints))
        if current_size <= 0:
            break
            
        print(f"Requesting {current_size} records (total fetched so far: {len(complaints)})...")
        params = {
            "has_narrative": "true",
            "size": current_size
        }
        if cursor:
            params["search_after"] = cursor
            
        response = requests.get(CFPB_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            print("No more complaints found.")
            break
            
        for hit in hits:
            source = hit.get("_source", {})
            record = {}
            for src_field, dest_field in fields_mapping.items():
                if src_field in source:
                    record[dest_field] = source[src_field]
            
            # Ensure the narrative is present and not empty
            narrative = record.get("consumer_complaint_narrative", "")
            if not narrative or pd.isna(narrative) or len(str(narrative).strip()) == 0:
                continue
                
            complaints.append(record)
            if len(complaints) >= sample_size:
                break
                
        # Set cursor to the sort values of the last hit
        last_hit = hits[-1]
        sort_vals = last_hit.get("sort")
        if sort_vals:
            cursor = "_".join(str(x) for x in sort_vals)
        else:
            print("No sort values found in hit. Cannot paginate further using cursor.")
            break
            
    df = pd.DataFrame(complaints)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} CFPB Consumer Complaints to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest datasets for Customer Intelligence Platform.")
    parser.add_argument("--bank", action="store_true", help="Ingest Bank Marketing dataset.")
    parser.add_argument("--cfpb", action="store_true", help="Ingest CFPB Consumer Complaints dataset.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for CFPB dataset.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw data.")
    
    args = parser.parse_args()
    
    # If neither --bank nor --cfpb is specified, default to running both
    run_all = not args.bank and not args.cfpb
    
    if run_all or args.bank:
        try:
            download_bank_marketing(args.output_dir)
        except Exception as e:
            print(f"Error ingesting Bank Marketing dataset: {e}")
            
    if run_all or args.cfpb:
        try:
            fetch_cfpb_complaints(args.output_dir, args.sample_size)
        except Exception as e:
            print(f"Error ingesting CFPB Consumer Complaints: {e}")
