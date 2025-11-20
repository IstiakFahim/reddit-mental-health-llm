# Your imports (unchanged)
import os
import re
import sys
import torch
import pandas as pd
import datetime
import shutil
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm
import difflib

# --- GPU Check ---
if not torch.cuda.is_available():
    print("ERROR: CUDA (GPU) is not available. Exiting.")
    sys.exit(1)
device = torch.device("cuda:0")
print("Using GPU:", torch.cuda.get_device_name(device))
sys.stdout.flush()

# --- Setup HF Cache Directories ---
local_tmp = os.environ.get("LOCAL_SCRATCH", "/tmp")
hf_cache_dir = os.path.join(local_tmp, "huggingface_cache")

os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["HF_METRICS_CACHE"] = hf_cache_dir

print(f"Hugging Face cache directory set to: {hf_cache_dir}")
sys.stdout.flush()

local_dir = os.path.join(local_tmp, "Emollama-7b")

# --- Input and output CSV paths ---
input_csv_path = "/scratch/project_2011211/Fahim/3dataset_part2.csv"
output_csv_path = os.path.join(local_tmp, "emollm_zeroshot_prediction_part2.csv")

# --- Lists of labels ---
CATEGORIES = """Mood Disorders
Anxiety Disorders
Obsessive-Compulsive & Related Disorders
Trauma & Stressor-Related Disorders
Personality Disorders
Neurodevelopmental Disorders
Schizophrenia Spectrum & Psychotic Disorders
Eating Disorders
Sleep Disorders
Substance-Related & Addictive Disorders"""

SUBCATEGORIES = """Depressive Disorders
Bipolar Disorders
Generalized Anxiety Disorder
Phobias
Panic Disorders
OCD Spectrum
PTSD Spectrum
Cluster A (Odd/Eccentric)
Cluster B (Dramatic/Emotional)
Cluster C (Anxious/Fearful)
Autism Spectrum Disorders
ADHD
Psychotic Disorders
Restrictive/Compensatory Disorders
Insomnia Spectrum
Substance Use Disorders"""

SPECIFIC_DISORDERS = """Adjustment Disorder
Agoraphobia
Alcohol Use Disorder
Anorexia Nervosa
Antisocial Personality Disorder
Attention-Deficit/Hyperactivity Disorder (ADHD)
Autism Spectrum Disorder (ASD)
Avoidant Personality Disorder
Binge-Eating Disorder
Bipolar 1 Disorder
Bipolar 2 Disorder
Body Dysmorphic Disorder
Bulimia Nervosa
Cannabis Use Disorder
Cyclothymic Disorder
Delusional Disorder
Dependent Personality Disorder
Generalized Anxiety Disorder (GAD)
Histrionic Personality Disorder
Hoarding Disorder
Insomnia
Major Depressive Disorder
Narcissistic Personality Disorder
Narcolepsy
Obsessive-Compulsive Disorder
Obsessive-Compulsive Personality Disorder
Panic Disorder
Panic Disorders
Paranoid Personality Disorder
Persistent Depressive Disorder (Dysthymia)
Post-Traumatic Stress Disorder (PTSD)
Restless Legs Syndrome
Schizoaffective Disorder
Schizoid Personality Disorder
Schizophrenia
Schizotypal Personality Disorder
Seasonal Affective Disorder
Sleep Apnea
Social Anxiety Disorder"""

allowed_categories = set(line.strip() for line in CATEGORIES.splitlines())
allowed_subcategories = set(line.strip() for line in SUBCATEGORIES.splitlines())
allowed_specific_disorders = set(line.strip() for line in SPECIFIC_DISORDERS.splitlines())


def truncate_repetition(text, max_length=5000):
    text = re.sub(r'(\b.+?\b)(\s+\1)+', r'\1', text)
    return text[:max_length]

def build_prompt(title, body):
    body = truncate_repetition(body)
    cat_list = ", ".join(line.strip() for line in CATEGORIES.splitlines())
    subcat_list = ", ".join(line.strip() for line in SUBCATEGORIES.splitlines())
    spec_list = ", ".join(line.strip() for line in SPECIFIC_DISORDERS.splitlines())

    return f"""You are a clinical psychology assistant.

Your task is to classify the Title and Body into EXACTLY one label from each list below.

Category (choose ONLY ONE from the following list, no other options):
{cat_list}

Subcategory (choose ONLY ONE from the following list, no other options):
{subcat_list}

Specific Disorder (choose ONLY ONE from the following list, no other options):
{spec_list}

IMPORTANT: Choose EXACTLY one label from the provided lists ONLY. Do NOT add explanations or additional text or repeat the instruction. If uncertain, reply with 'UNKNOWN'.
Respond EXACTLY as:

Category:
Subcategory:
Specific Disorder:
---

Title: {title}
Body: {body}
"""


# --- Load input ---
print(f"Loading data from {input_csv_path} ...")
df = pd.read_csv(input_csv_path)

# Ensure 'Title' and 'Body' are strings
df['Title'] = df['Title'].astype(str)
df['Body'] = df['Body'].astype(str)

# --- GPU list ---
print("Available GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
sys.stdout.flush()

# --- Download model ---
print("Downloading model snapshot...")
sys.stdout.flush()
snapshot_download(repo_id="lzw1008/Emollama-7b", local_dir=local_dir)
print("Files in model directory after download:", os.listdir(local_dir))
sys.stdout.flush()

# --- Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
tokenizer.padding_side = "left"
tokenizer.parallelism = True
print("Tokenizer loaded.")

# --- Model ---
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    trust_remote_code=True,
    local_files_only=True,
    use_safetensors=True,
    device_map='auto'
)
model.eval()
print("Model loaded.")

# --- Inference helpers ---
def get_generate_fn(model):
    return model.module.generate if hasattr(model, "module") else model.generate

def run_batch_prompts(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generate_fn = get_generate_fn(model)

    with torch.no_grad():
        outputs = generate_fn(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
            
        )

    decoded_outputs = []
    for i, output in enumerate(outputs):
        full = tokenizer.decode(output, skip_special_tokens=True)
        # print(f"\n[MODEL OUTPUT for prompt {i+1}]\n{full}\n")  # <-- Print raw model output here

        start = full.find("Category:")
        trimmed = full[start:].strip() if start != -1 else full.strip()
        decoded_outputs.append(trimmed)
    return decoded_outputs

def closest_match(value, allowed_set):
    matches = difflib.get_close_matches(value, allowed_set, n=1, cutoff=0.8)
    return matches[0] if matches else None

def extract_predictions(text):
    lines = text.strip().splitlines()
    fields = {"Category": "UNKNOWN", "Subcategory": "UNKNOWN", "Specific Disorder": "UNKNOWN"}

    for line in lines:
        for key in fields:
            if line.strip().lower().startswith(key.lower() + ":"):
                value = line.split(":", 1)[1].strip()

                if key == "Category":
                    match = closest_match(value, allowed_categories)
                    fields[key] = match if match else "UNKNOWN"
                elif key == "Subcategory":
                    match = closest_match(value, allowed_subcategories)
                    fields[key] = match if match else "UNKNOWN"
                elif key == "Specific Disorder":
                    match = closest_match(value, allowed_specific_disorders)
                    fields[key] = match if match else "UNKNOWN"
    return fields["Category"], fields["Subcategory"], fields["Specific Disorder"]


# --- Classification logic ---
def classify_dataframe_in_batches(df, tokenizer, model, batch_size=8):
    categories, subcategories, specific_disorders = [], [], []
    total = len(df)
    unknown_count = 0
    failed_cases = []

    for start_idx in tqdm(range(0, total, batch_size), desc="Classifying"):
        torch.cuda.empty_cache()
        gc.collect()

        batch_df = df.iloc[start_idx:start_idx + batch_size]
        prompts = [build_prompt(row['Title'], row['Body']) for _, row in batch_df.iterrows()]

        try:
            outputs = run_batch_prompts(prompts, tokenizer, model)
        except Exception as e:
            print(f"Error during inference batch starting at row {start_idx}: {e}")
            outputs = [""] * len(prompts)

        for output, (_, row) in zip(outputs, batch_df.iterrows()):
            cat, subcat, spec = extract_predictions(output)
            categories.append(cat)
            subcategories.append(subcat)
            specific_disorders.append(spec)
            if "UNKNOWN" in (cat, subcat, spec):
                failed_cases.append((row['Title'], row['Body'], output))
                unknown_count += 1

        print(f"[{datetime.datetime.now()}] Processed rows {start_idx + 1} to {min(start_idx + batch_size, total)}")
        sys.stdout.flush()

    df['Predicted Category'] = categories
    df['Predicted Subcategory'] = subcategories
    df['Predicted Specific Disorder'] = specific_disorders

    print(f"\nTotal UNKNOWN predictions: {unknown_count} out of {total}")

    if failed_cases:
        with open(os.path.join(local_tmp, "failed_predictions.txt"), "w", encoding="utf-8") as f:
            for title, body, output in failed_cases:
                f.write(f"---\nTitle: {title}\nBody: {body}\nModel Output:\n{output}\n\n")

    return df

# --- Run inference ---
df_with_preds = classify_dataframe_in_batches(df, tokenizer, model, batch_size=8)

# --- Save output ---
df_with_preds.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")
os.system(f"cp {output_csv_path} /scratch/project_2011211/Fahim/")

# --- Cleanup ---
print("Cleaning up Hugging Face cache and model directories...")
shutil.rmtree(hf_cache_dir, ignore_errors=True)
shutil.rmtree(local_dir, ignore_errors=True)
print("Cleanup done.")
