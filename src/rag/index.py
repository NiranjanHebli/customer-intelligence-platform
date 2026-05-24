import os
import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/cfpb_complaints.csv"
INDEX_DIR = "src/rag"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text: str, size: int, overlap: int):
    """Simple sliding window character chunker."""
    if not isinstance(text, str) or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks

def build_index():
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Drop rows without narratives
    df = df.dropna(subset=['consumer_complaint_narrative'])
    logger.info(f"Found {len(df)} complaints with narratives.")
    
    # Process and chunk
    logger.info("Chunking texts...")
    documents = []
    metadata = []
    
    for idx, row in df.iterrows():
        narrative = row['consumer_complaint_narrative']
        complaint_id = row['complaint_id']
        product = row['product']
        issue = row['issue']
        
        chunks = chunk_text(narrative, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "chunk_id": f"{complaint_id}_{i}",
                "complaint_id": complaint_id,
                "product": product,
                "issue": issue,
                "text": chunk
            })
            
    logger.info(f"Generated {len(documents)} chunks. Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info("Encoding documents (this may take a few minutes)...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    logger.info(f"Saving index to {INDEX_PATH} and metadata to {METADATA_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    # Convert metadata to a dict mapped by integer ID (FAISS returns integer indices)
    metadata_map = {i: meta for i, meta in enumerate(metadata)}
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata_map, f)
        
    logger.info("Indexing complete.")

if __name__ == "__main__":
    build_index()
