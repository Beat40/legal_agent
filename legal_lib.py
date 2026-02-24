
import torch
import faiss
import pandas as pd
import numpy as np
import os
import time
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config & Setup ---
QA_MODEL_NAME = "alex-apostolo/legal-bert-base-cuad"
EMBEDDER_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_PATH = r"C:\Users\sengu\agentic\legal_agent\vectore_db\eurlex_mpnet.index"
CHUNKS_PATH = r"C:\Users\sengu\agentic\legal_agent\vectore_db\eurlex_chunks.parquet"

# CUAD Schema
CUAD_QA_SCHEMA = {
    "Assignment": "Does this contract specify an assignment clause?",
    "Anti-Assignment": "Does this contract restrict assignment?",
    "Change of Control": "Does this contract specify a change of control clause?",
    "Subcontracting": "Does this contract specify a subcontracting clause?",
    "Third Party Beneficiaries": "Does this contract specify third party beneficiaries?",
    "Joint and Several Liability": "Does this contract specify joint and several liability?",
    "Termination for Convenience": "Does this contract specify a termination for convenience clause?",
    "Termination for Cause": "Does this contract specify a termination for cause clause?",
    "Contract Duration": "Does this contract specify a contract duration?",
    "Renewal Term": "Does this contract specify a renewal term?",
    "Notice Period": "Does this contract specify a notice period?",
    "Limitation of Liability": "Does this contract specify a limitation of liability?",
    "Cap on Liability": "Does this contract specify a cap on liability?",
    "Consequential Damages": "Does this contract exclude consequential damages?",
    "Liquidated Damages": "Does this contract specify liquidated damages?",
    "Indemnification": "Does this contract specify an indemnification clause?",
    "Covenant Not To Sue": "Does this contract specify a covenant not to sue?",
    "Confidentiality": "Does this contract specify a confidentiality clause?",
    "Return of Materials": "Does this contract specify a return of materials clause?",
    "Data Protection": "Does this contract specify a data protection clause?",
    "Intellectual Property": "Does this contract specify intellectual property ownership?",
    "Governing Law": "Does this contract specify a governing law?",
    "Dispute Resolution": "Does this contract specify a dispute resolution clause?",
    "Compliance with Laws": "Does this contract specify compliance with laws?",
    "Insurance": "Does this contract specify insurance requirements?",
    "Audits": "Does this contract specify audit rights?",
    "Payment Terms": "Does this contract specify payment terms?",
    "Performance Standards": "Does this contract specify performance standards?",
    "Most Favoured Nation": "Does this contract include a most favoured nation clause?",
    "Non Compete": "Does this contract include a non-compete clause?",
    "Non Disparagement": "Does this contract include a non-disparagement clause?",
    "Exclusivity": "Does this contract specify exclusivity?",
    "Restrictive Covenants": "Does this contract specify restrictive covenants?",
    "Anti-Embarrassment": "Does this contract include an anti-embarrassment clause?",
    "Force Majeure": "Does this contract specify a force majeure clause?",
    "Waiver": "Does this contract specify a waiver clause?",
    "Severability": "Does this contract specify a severability clause?",
    "Representations and Warranties": "Does this contract specify representations and warranties?",
    "Right of First Refusal": "Does this contract specify a right of first refusal?",
    "Anti-Sandbagging": "Does this contract include an anti-sandbagging clause?",
    "Anti-Assignment (Narrow)": "Does this contract include a narrowly tailored anti-assignment clause?"
}

def load_models():
    """Initializes and returns the QA model, tokenizer, and embedder."""
    print("Loading models...")
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    if torch.cuda.is_available():
        qa_model = qa_model.cuda().half() # Use FP16 for speed
    qa_model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    return qa_model, qa_tokenizer, embedder

def load_data():
    """Loads the FAISS index and EUR-Lex chunks."""
    print("Loading FAISS index and data...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Index or Chunks file not found at {INDEX_PATH} or {CHUNKS_PATH}")
        
    index = faiss.read_index(INDEX_PATH)
    chunks_df = pd.read_parquet(CHUNKS_PATH)
    return index, chunks_df

def init_llm():
    """Initializes the LLM client."""
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# --- Helper Functions ---

def chunk_contract(text, tokenizer, max_tokens=400, overlap=50):
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end]
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

def cache_cuad_questions(tokenizer, schema, device="cuda"):
    questions = list(schema.values())
    q_tokens = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True)
    if torch.cuda.is_available() and device == "cuda":
        q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
    return {"labels": list(schema.keys()), "questions": questions, "q_tokens": q_tokens}

def qa_based_classification_batched_optimized(model, tokenizer, context, cached_questions, device="cuda", max_answer_len=50, top_k=20):
    # This matches the optimized implementation from refined_flow.ipynb
    timings = {}
    t0 = time.time()

    labels = cached_questions["labels"]
    q_tokens = cached_questions["q_tokens"]
    num_labels = len(labels)

    # --- Input building (vectorized) ---
    t2 = time.time()
    context_tokens = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    
    if torch.cuda.is_available() and device == "cuda":
        context_tokens = context_tokens.to(device)

    # Get context offsets (skip first CLS token)
    context_offsets = context_tokens["offset_mapping"][0][1:].cpu().numpy()

    input_ids = []
    attention_masks = []
    q_lengths = [] 

    for i in range(num_labels):
        q_ids = q_tokens["input_ids"][i]
        q_lengths.append(len(q_ids))
        
        ids = torch.cat([q_ids, context_tokens["input_ids"][0][1:]])[:512]
        input_ids.append(ids)
        attention_masks.append(torch.ones_like(ids))

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    timings["input_build"] = time.time() - t2

    # --- Forward pass (FP16) ---
    t3 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
    timings["forward"] = time.time() - t3

    # --- FULL Vectorized span search (Across all labels at once) ---
    t4 = time.time()
    results = {}

    s_logits = outputs.start_logits.float() # [41, 512]
    e_logits = outputs.end_logits.float()   # [41, 512]
    s_logits[:, 0] = -1e9
    e_logits[:, 0] = -1e9
    
    seq_len = s_logits.size(1)

    # Broadcated score matrix for all 41 labels: [41, 512, 512]
    # rows = start, cols = end
    scores = s_logits.unsqueeze(2) + e_logits.unsqueeze(1)

    # 1. Mask invalid: end < start
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
    scores = scores * mask + (1 - mask) * -1e9

    # 2. Mask invalid: len > max_answer_len
    len_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=max_answer_len).unsqueeze(0)
    scores = scores * len_mask + (1 - len_mask) * -1e9

    # 3. Batch Argmax
    # Flatten last two dims and find argmax for each label
    flat_scores = scores.view(num_labels, -1)
    best_indices = flat_scores.argmax(dim=1)
    best_scores = flat_scores.gather(1, best_indices.unsqueeze(1)).squeeze(1)

    start_indices = best_indices // seq_len
    end_indices = best_indices % seq_len

    # Map back to labels
    for i in range(num_labels):
        label = labels[i]
        b_score = best_scores[i].item()
        
        if b_score > 5.0:
            s_idx = start_indices[i].item()
            e_idx = end_indices[i].item()
            q_len = q_lengths[i]
            
            if s_idx >= q_len:
                rel_s, rel_e = s_idx - q_len, e_idx - q_len
                if rel_e < len(context_offsets):
                    answer = context[context_offsets[rel_s][0] : context_offsets[rel_e][1]]
                else:
                    answer = tokenizer.decode(input_ids[i][s_idx : e_idx+1], skip_special_tokens=True)
            else:
                answer = tokenizer.decode(input_ids[i][s_idx : e_idx+1], skip_special_tokens=True)
            
            if len(answer.split()) >= 6:
                results[label] = {"present": True, "text": answer, "confidence": float(b_score)}
            else:
                results[label] = {"present": False, "text": None, "confidence": float(b_score)}
        else:
            results[label] = {"present": False, "text": None, "confidence": float(b_score)}

    timings["span_search"] = time.time() - t4
    timings["total"] = time.time() - t0
    return results, timings

def run_contract_analysis_sequential(chunks, model, tokenizer, cached_questions, device="cuda", callback=None):
    all_results = []
    timing_stats = []

    for i, chunk in enumerate(tqdm(chunks, desc="BERT Extraction")):
        qa_results, timings = qa_based_classification_batched_optimized(
            model=model,
            tokenizer=tokenizer,
            context=chunk,
            cached_questions=cached_questions,
            device=device
        )

        timing_stats.append(timings)
        
        # Periodic callback to update UI/VRAM
        if callback and i % 5 == 0:
            callback()

        for label, res in qa_results.items():
            if not res["present"]:
                continue

            all_results.append({
                "chunk_id": i,
                "label": label,
                "confidence": res["confidence"],
                "span": res["text"]
            })

    return all_results, timing_stats

# --- LLM Labeling ---
def label_span(span, llm_client):
    prompt = f"""
    Analyze this contract clause and identify its category from the following list (CUAD categories). 
    If it doesn't fit well, use 'General'. Returns ONLY the category name.
    
    Categories: {', '.join(CUAD_QA_SCHEMA.keys())}
    
    Clause: "{span}"
    
    Category:
    """
    try:
        response = llm_client.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error labeling span: {e}")
        return "General"

def label_spans_with_llm_parallel(spans, llm_client, max_workers=5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {executor.submit(label_span, span, llm_client): span for span in spans}
        for future in tqdm(as_completed(future_to_span), total=len(spans), desc="Labeling Spans"):
            span = future_to_span[future]
            try:
                results[span] = future.result()
            except Exception as e:
                print(f"Error labeling span: {e}")
                results[span] = "General"
    return results

# --- Retrieval ---
def retrieve_eurlex_evidence(span, label, embedder, index, chunks_df, k=5):
    query = f"EU law regarding {label} clauses: {span}"
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, k)
    
    evidence = []
    for score, idx in zip(scores[0], indices[0]):
        row = chunks_df.iloc[idx]
        evidence.append({
            "celex_id": row["celex_id"],
            "title": row["title"],
            "text": row["text"],
            "score": float(score)
        })
    return evidence

# --- Reasoning ---
def verify_clause_compliance(span, label, evidence, llm_client):
    evidence_text = "\n\n".join([f"Law: {e['title']} (ID: {e['celex_id']})\nText: {e['text']}" for e in evidence])
    prompt = f"""
    You are a legal AI assistant. Analyze the compliance of the following contract clause with the provided EU laws.
    
    Contract Clause ({label}): 
    "{span}"
    
    Relevant EU Laws:
    {evidence_text}
    
    Analyze:
    1. Does the clause align with EU regulations?
    2. Are there any potential risks or conflicts?
    3. Provide a brief compliance status (Compliant/Non-Compliant/Needs Review).
    
    Response:
    """
    try:
        response = llm_client.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return "Analysis Failed"

def process_span_full(idx, span, label, embedder, index, chunks_df, llm_client):
    """Combines retrieval and reasoning for a single span to facilitate parallel execution."""
    # 1. Retrieve
    evidence = retrieve_eurlex_evidence(span, label, embedder, index, chunks_df)
    
    # 2. Reason
    analysis = verify_clause_compliance(span, label, evidence, llm_client)
    
    return {
        "id": idx,
        "span": span,
        "label": label,
        "evidence": evidence,
        "analysis": analysis,
        "status": "Review Pending",
        "comment": ""
    }
