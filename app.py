import subprocess
import sys

try:
    import duckduckgo_search
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "duckduckgo-search"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

import os
import warnings

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"

sys.stderr = open(os.devnull, 'w')

import torch
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()
logging.disable_progress_bar()

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def ask_qwen(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs, max_new_tokens=500, temperature=0.3)
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def search_and_extract(query):
    extracted_texts = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            for r in results:
                try:
                    res = httpx.get(r['href'], headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0, follow_redirects=True)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.extract()
                    clean_text = soup.get_text(separator=' ', strip=True)[:1000]
                    extracted_texts.append(clean_text)
                except:
                    extracted_texts.append(r['body'])
    except:
        pass
    return "\n".join(extracted_texts)

while True:
    try:
        user_query = input()
    except (KeyboardInterrupt, EOFError):
        break

    if not user_query.strip():
        continue

    goals_prompt = (
        "You are an autonomous research AI. Create a bulleted checklist of exactly 3 core verification goals "
        "necessary to answer the user query accurately and comprehensively.\n"
        "STRICT INSTRUCTION: Focus entirely on the exact language used by the user. "
        "Generate your response by concentrating exclusively on that single language."
    )
    goals_text = ask_qwen(goals_prompt, f"User Input: {user_query}\n\nGenerate the research goals.")
    goals_list = [g.strip() for g in goals_text.split('\n') if g.strip()][:3]

    raw_data_combined = []

    for goal in goals_list:
        query_prompt = (
            "Convert the research goal into an unambiguous English web search query string.\n"
            "Identify the canonical entity name or primary concept behind the goal, resolving any phonetic "
            "transliterations, informal spellings, or non-Latin scripts to their standard official terms.\n"
            "Output ONLY the search query string."
        )
        search_query = ask_qwen(query_prompt, f"Original User Prompt: {user_query}\nGoal: {goal}")
        
        raw_info = search_and_extract(search_query)
        raw_data_combined.append(f"Goal: {goal}\nInformation: {raw_info}")

    all_context = "\n\n".join(raw_data_combined)

    final_prompt = (
        "You are an expert factual research assistant. Synthesize the research findings to answer the user prompt directly in EXACTLY ONE sentence.\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Identify and resolve any phonetic, informal, or transliterated entity names to their official canonical real-world identity.\n"
        "2. Base your response on factual accuracy and world knowledge, filtering out irrelevant or noisy search snippets.\n"
        "3. Focus entirely on the language spoken by the user and generate the response while remaining strictly dedicated to that language."
    )
    final_summary = ask_qwen(final_prompt, f"User Prompt: {user_query}\n\nSearch Context:\n{all_context}")

    print(final_summary)
