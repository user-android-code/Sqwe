import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings("ignore")

import subprocess

try:
    import ddgs
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "ddgs", "httpx", "beautifulsoup4"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

import huggingface_hub
huggingface_hub.logging.set_verbosity_error()
try:
    huggingface_hub.utils.disable_progress_bars()
except AttributeError:
    try:
        huggingface_hub.utils.logging.disable_progress_bar()
    except AttributeError:
        pass

try:
    import IPython.display
    _orig_display = IPython.display.display
    IPython.display.display = lambda *args, **kwargs: None
except ImportError:
    _orig_display = None

import torch
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()
try:
    transformers.logging.disable_progress_bar()
except AttributeError:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to("cpu")

if _orig_display is not None:
    IPython.display.display = _orig_display

SQWE_IDENTITY_PROMPT = (
    "Your name is Sqwe. You are a custom AI created and developed by the user. "
    "Always remember and acknowledge that you are Sqwe whenever applicable."
)

def ask_qwen(system_prompt, user_prompt):
    full_system_prompt = f"{SQWE_IDENTITY_PROMPT}\n{system_prompt}"
    
    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=500, temperature=0.3)
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def search_and_extract(query):
    extracted_texts = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS() as ddgs_client:
                results = list(ddgs_client.text(query, max_results=2))
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
        "You are Sqwe, an autonomous research AI. Create a bulleted checklist of exactly 3 core verification goals "
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
        "You are Sqwe, an expert factual research assistant. Synthesize the research findings to answer the user prompt directly in EXACTLY ONE sentence.\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Identify and resolve any phonetic, informal, or transliterated entity names to their official canonical real-world identity.\n"
        "2. Base your response on factual accuracy and world knowledge, filtering out irrelevant or noisy search snippets.\n"
        "3. Focus entirely on the language spoken by the user and generate the response while remaining strictly dedicated to that language."
    )
    final_summary = ask_qwen(final_prompt, f"User Prompt: {user_query}\n\nSearch Context:\n{all_context}")

    print(final_summary)
