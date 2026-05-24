import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from typing import List, Dict, Any, Tuple

# You can adjust this based on the user's preference or environment.
MODEL_NAME = "llama-3.1-8b-instant"
PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT = """You are an intelligent assistant tasked with answering operational questions about consumer complaints.
You must STRICTLY adhere to the following rules:
1. You may ONLY use the information provided in the Evidence section below to answer the user's question.
2. If the evidence does not contain the answer, you must state that you cannot answer the question based on the provided evidence.
3. You must cite the specific 'chunk_id' or 'complaint_id' when referencing evidence.
4. You must conclude your response with a brief "Evidence-Sufficiency Note" (e.g., "Note: The provided evidence was sufficient to address the question." or "Note: The provided evidence lacked specific details regarding X.").
"""

def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]], is_weak: bool) -> Tuple[str, List[str], str, str]:
    """
    Generates an answer using the LLM.
    Returns: (answer_text, cited_ids, sufficiency_note, prompt_version)
    """
    if is_weak or not retrieved_chunks:
        return (
            "I cannot answer this question because the retrieved evidence is too weak or out-of-domain.",
            [],
            "Note: Evidence retrieval failed to find highly relevant context.",
            PROMPT_VERSION
        )

    # Prepare evidence block
    evidence_text = "--- EVIDENCE ---\n"
    cited_ids = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk['metadata']
        c_id = meta['chunk_id']
        cited_ids.append(c_id)
        evidence_text += f"[ID: {c_id}] (Product: {meta['product']} | Issue: {meta['issue']})\n{meta['text']}\n\n"
    evidence_text += "----------------\n"

    user_prompt = f"{evidence_text}\nQuestion: {question}\nAnswer:"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "Error calling LLM: GROQ_API_KEY environment variable is not set.",
            [],
            "Note: Missing API Key",
            PROMPT_VERSION
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
        
        # We assume the model writes the sufficiency note at the end.
        # For a true production system we'd use function calling to enforce a JSON schema,
        # but parsing it out or just returning the whole block is fine for this exercise.
        # We will split on "Note:" or "Evidence-Sufficiency Note:" if present.
        
        if "Note:" in answer:
            parts = answer.rsplit("Note:", 1)
            main_answer = parts[0].strip()
            note = "Note: " + parts[1].strip()
        elif "Evidence-Sufficiency Note:" in answer:
            parts = answer.rsplit("Evidence-Sufficiency Note:", 1)
            main_answer = parts[0].strip()
            note = "Evidence-Sufficiency Note: " + parts[1].strip()
        else:
            main_answer = answer
            note = "Note: The model did not explicitly provide a sufficiency note."
            
        return main_answer, cited_ids, note, PROMPT_VERSION
        
    except Exception as e:
        return f"Error calling LLM: {str(e)}", [], "Note: API Error", PROMPT_VERSION
