import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_FILE = "data.txt"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def build_vector_db():
    if not os.path.exists(DATA_FILE):
        return None, []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+300] for i in range(0, len(text), 250) if len(text[i:i+300]) > 50]

    if not chunks:
        return None, []

    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, chunks

index, chunks = build_vector_db()

def search_context(query, top_k=2):
    if index is None or len(chunks) == 0:
        return ""

    query_vector = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)

    retrieved_texts = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    return "\n---\n".join(retrieved_texts)

SYSTEM_PROMPT = """[System Instruction]
Your name is "sqwe".
You are an AI assistant powered by the provided context (extracted from data.txt) to engage in conversation with users.

[Language Rules]
- Default Language: You MUST ALWAYS respond in English by default.
- Language Switch: Only switch and reply in Japanese if the user explicitly requests you to speak in Japanese (e.g., "日本語でしゃべって", "日本語で話して").

[Rules & Restrictions]
1. Identity:
   - Always remember that your name is "sqwe". If asked for your name, reply as "sqwe".

2. Violation Detection:
   - Analyze both the user's input and the provided context.
   - If you detect any requests or topics related to hacking, cyber attacks, unauthorized access, vulnerability exploits, password cracking, malware creation, or any illegal/dangerous activities:
   - Do NOT provide any explanations, excuses, or extra text.

3. Violation Response:
   - Immediately and strictly output ONLY the following exact English sentence:

"That phrase (or topic) may be in violation of the Terms of Service."
"""

def ask_qwen(player_input):
    context = search_context(player_input)

    user_content = f"""【Context】
{context}

【User Input】
{player_input}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    while True:
        user_msg = input("\nPlayer: ")
        if user_msg.lower() in ["exit", "quit"]:
            break
        
        ans = ask_qwen(user_msg)
        print(f"AI: {ans}")
