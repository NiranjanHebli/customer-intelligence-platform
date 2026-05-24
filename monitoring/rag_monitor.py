import time
import requests
import statistics

API_URL = "http://127.0.0.1:8000/ask-complaints"

test_queries = [
    "Why was my credit card blocked?",
    "Are there hidden fees for maintaining a checking account?",
    "How can I dispute a transaction on my credit report?",
    "Tell me about mortgage loan transferring issues.",
    "What is the best recipe for baking a chocolate cake?" # Out of domain query
]

def run_rag_monitor():
    latencies = []
    refusals = 0
    total = len(test_queries)

    print(f"Running RAG monitor script against {API_URL}...")
    
    for idx, query in enumerate(test_queries, 1):
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"question": query}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "").lower()
            cited_ids = data.get("cited_evidence_ids", [])
            
            refusal_keywords = ["cannot answer", "cannot provide", "do not have", "cannot find"]
            is_refusal = any(k in answer for k in refusal_keywords) or len(cited_ids) == 0
            if is_refusal:
                refusals += 1
                
        except requests.exceptions.RequestException as e:
            print(f"Request {idx} failed (server might not be running): {e}")
            refusals += 1
            
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    refusal_rate = (refusals / total) * 100 if total > 0 else 0.0

    print("\n--- RAG Monitoring Metrics ---")
    print(f"Total Test Queries: {total}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Refusals: {refusals}")
    print(f"Refusal Rate: {refusal_rate:.2f}%")
    print("------------------------------")

if __name__ == "__main__":
    run_rag_monitor()
