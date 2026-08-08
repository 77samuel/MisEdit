# ============================================================
# MisEdit — TruthfulQA Cross-Dataset Validation (v8)
# Single standalone cell. ROME + WISE on 48 TruthfulQA items.
# Checkpoint: /kaggle/working/MisEdit_TruthfulQA_ROME_WISE.csv
# ============================================================

import subprocess, sys, os, gc, json, torch, yaml, shutil, warnings
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

# ── Step 1: Install dependencies ──────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install",
    "einops", "hydra-core", "higher", "omegaconf",
    "sentence-transformers", "peft", "sentencepiece",
    "rouge", "datasets", "fairscale", "zhipuai",
    "statsmodels", "iopath", "av", "qwen-vl-utils", "-q"], check=True)
print("Dependencies installed.")

# ── Step 2: Clone and patch EasyEdit ──────────────────────────
subprocess.run(["git", "clone", "https://github.com/zjunlp/EasyEdit.git",
                "/kaggle/working/EasyEdit"], capture_output=True)
sys.path.insert(0, '/kaggle/working/EasyEdit')
os.chdir('/kaggle/working/EasyEdit')

f1 = "easyeditor/models/rome/layer_stats.py"
c1 = open(f1).read()
if '20200501.en' in c1:
    c1 = c1.replace(
        'dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]',
        'dict(wikitext="wikitext-103-raw-v1", wikipedia="wikitext-103-raw-v1")[ds_name]'
    )
    open(f1, 'w').write(c1)
    print("EasyEdit patch applied.")
else:
    print("EasyEdit patch already applied.")

# ── Step 3: Load TruthfulQA MC1 ───────────────────────────────
from datasets import load_dataset

ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
items_raw = list(ds['validation'])
print(f"TruthfulQA loaded: {len(items_raw)} raw items")

# ── Step 4: Delete stale files and rebuild selection ──────────
SAVE_PATH     = "/kaggle/working/MisEdit_TruthfulQA_ROME_WISE.csv"
SELECTED_PATH = "/kaggle/working/MisEdit_TruthfulQA_Selected48.json"

for path in [SAVE_PATH, SELECTED_PATH]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")

import random
random.seed(42)

candidates = []
for i, item in enumerate(items_raw):
    choices = item['mc1_targets']['choices']
    labels  = item['mc1_targets']['labels']
    cat     = item.get('category', item.get('type', 'TruthfulQA'))
    if len(choices) >= 2:
        correct_idx = labels.index(1) if 1 in labels else 0
        clean_label = (item['question'][:50]
                       .replace(' ', '_').replace('?', '')
                       .replace(',', '').replace("'", "").replace('"', ''))
        candidates.append({
            'truthfulqa_id': i,
            'label':         clean_label,
            'question':      item['question'],
            'category':      cat,
            'choices':       choices,
            'correct_idx':   correct_idx,
            'num_options':   len(choices)
        })

selected = random.sample(candidates, min(48, len(candidates)))
with open(SELECTED_PATH, 'w') as f:
    json.dump(selected, f, indent=2)
print(f"Selected {len(selected)} items frozen.")

# ── Step 5: Helper functions ───────────────────────────────────
MAX_LEN = 512

def get_logits(model, input_ids):
    """
    Get logits safely from any model type.
    Handles: CausalLM, ROME-wrapped, base model outputs.
    """
    out = model(input_ids=input_ids)
    # Try .logits first (CausalLMOutputWithPast)
    if hasattr(out, 'logits'):
        return out.logits[0]
    # BaseModelOutputWithPast — need to pass through lm_head
    hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
    # Try to find lm_head
    lm_head = None
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        lm_head = model.model.lm_head
    if lm_head is not None:
        return lm_head(hidden)[0]
    raise ValueError("Cannot extract logits from model output")

def score_mc_normalized(model, tokenizer, question, choices, correct_idx):
    """
    Score each MC option by average log-prob of option text given question.
    Normalized to 0-1 using softmax — invariant to number of choices.
    Truncates to MAX_LEN tokens.
    """
    raw_scores = []
    q_enc = tokenizer(
        f"{question}\nAnswer:",
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True
    )
    q_ids = q_enc['input_ids'].to('cuda:0')
    q_len = q_ids.shape[1]

    for text in choices:
        full_enc = tokenizer(
            f"{question}\nAnswer: {text}",
            return_tensors="pt",
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation=True
        )
        full_ids = full_enc['input_ids'].to('cuda:0')
        opt_len  = full_ids.shape[1] - q_len
        if opt_len <= 0:
            raw_scores.append(-999.0)
            continue
        with torch.no_grad():
            logits = get_logits(model, full_ids)
        lp    = F.log_softmax(logits, dim=-1)
        start = q_len - 1
        score = sum(
            lp[start+i, full_ids[0, q_len+i].item()].item()
            for i in range(opt_len)
        ) / opt_len
        raw_scores.append(score)

    raw_arr = np.array(raw_scores)
    valid   = raw_arr > -999
    norm    = np.zeros(len(choices))
    if valid.sum() > 0:
        exp = np.exp(raw_arr[valid] - raw_arr[valid].max())
        norm[valid] = exp / exp.sum()

    pred_idx = int(np.argmax(norm))
    return {
        'predicted_idx':      pred_idx,
        'correct_score_norm': float(norm[correct_idx]),
        'predicted_correct':  pred_idx == correct_idx,
    }

def extract_subject(question):
    words = question.split()
    skip  = {"Is","Did","Do","Does","Can","Are","Was","Were",
             "Have","Has","What","Why","How","Which","Who"}
    start = 1 if words[0] in skip else 0
    for length in range(min(6, len(words)-start), 0, -1):
        candidate = " ".join(words[start:start+length])
        if candidate in question:
            return candidate
    return words[start] if len(words) > start else words[0]

# ── Step 6: Write hparams ─────────────────────────────────────
from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams

rome_dst = "hparams/ROME/LlamaForCausalLM.yaml"
shutil.copy("hparams/ROME/llama3.2-3b.yaml", rome_dst)
with open(rome_dst) as f: cfg = yaml.safe_load(f)
cfg.update({
    'model_name':       'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'device':           0,
    'mom2_adjustment':  False,
    'v_num_grad_steps': 10,
    'v_loss_layer':     21,
})
with open(rome_dst, 'w') as f: yaml.dump(cfg, f)

wise_dst = "hparams/WISE/LlamaForCausalLM.yaml"
shutil.copy("hparams/WISE/llama3.2-3b.yaml", wise_dst)
with open(wise_dst) as f: cfg = yaml.safe_load(f)
cfg.update({
    'model_name':   'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'device':       0,
    'inner_params': ['model.layers[21].mlp.down_proj.weight']
})
for key in ['v_num_grad_steps','v_loss_layer','mom2_adjustment',
            'mom2_update_weight','layers','sequential_edit']:
    cfg.pop(key, None)
with open(wise_dst, 'w') as f: yaml.dump(cfg, f)
print("Hparams written.")

# ── Step 7: Load model and editors ────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LEN

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16,
    device_map="cuda:0", trust_remote_code=True)
torch.cuda.empty_cache(); gc.collect()
print(f"Model loaded. GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# Test logits extraction before running experiments
test_ids = tokenizer("Hello world", return_tensors="pt")['input_ids'].to('cuda:0')
with torch.no_grad():
    test_logits = get_logits(model, test_ids)
print(f"Logits test: shape={test_logits.shape} — OK")

rome_hparams = ROMEHyperParams.from_hparams(rome_dst)
rome_hparams.model_parallel = False
rome_hparams.device = 0
rome_editor = BaseEditor.from_hparams(rome_hparams)
print("ROME editor ready.")

wise_hparams = WISEHyperParams.from_hparams(wise_dst)
wise_hparams.device = 0
wise_hparams.sequential_edit = False
wise_editor = BaseEditor.from_hparams(wise_hparams)
print("WISE editor ready.")

# ── Step 8: Load checkpoint ────────────────────────────────────
rows      = []
done_keys = set()

if os.path.exists(SAVE_PATH):
    existing  = pd.read_csv(SAVE_PATH)
    rows      = existing.to_dict('records')
    done_keys = set(zip(existing['label'], existing['method']))
    print(f"Checkpoint: {len(done_keys)} items already done.")
else:
    print("No checkpoint. Starting fresh.")

# ── Step 9: Run ROME then WISE ────────────────────────────────
METHODS = [('ROME', rome_editor), ('WISE', wise_editor)]
total   = len(selected) * 2
done_n  = len(done_keys)

for method_name, editor in METHODS:
    print(f"\n{'='*55}")
    print(f"{method_name} — {len(selected)} TruthfulQA items")
    print(f"{'='*55}")

    for item in selected:
        key = (item['label'], method_name)
        if key in done_keys:
            continue

        question    = item['question']
        choices     = item['choices']
        corr_idx    = item['correct_idx']
        subject     = extract_subject(question)
        target      = choices[corr_idx]
        tqa_id      = item['truthfulqa_id']
        category    = item['category']
        num_options = item['num_options']

        try:
            pre = score_mc_normalized(model, tokenizer,
                                      question, choices, corr_idx)

            if method_name == 'ROME':
                metrics, edited_model, _ = editor.edit(
                    prompts=[question],
                    rephrase_prompts=[question],
                    target_new=[target],
                    subject=[subject],
                    keep_original_weight=True
                )
            else:
                metrics, edited_model, _ = editor.edit(
                    prompts=[question],
                    rephrase_prompts=[question],
                    target_new=[target],
                    subject=[subject],
                    keep_original_weight=True,
                    loc_prompts=["nq question: what is the capital of France Paris"]
                )

            rw_pre  = float(metrics[0]['pre']['rewrite_acc'][0])
            rw_post = float(metrics[0]['post']['rewrite_acc'][0])

            # Unwrap edited model to get one with .logits support
            # ROME: edited_model is patched CausalLM directly
            # WISE: edited_model.model is the inner CausalLM
            if hasattr(edited_model, 'lm_head'):
                inner = edited_model
            elif hasattr(edited_model, 'model') and hasattr(edited_model.model, 'lm_head'):
                inner = edited_model.model
            else:
                inner = edited_model  # fallback
            post = score_mc_normalized(inner, tokenizer,
                                       question, choices, corr_idx)

            del edited_model, inner
            torch.cuda.empty_cache(); gc.collect()

            mc_delta   = round(post['correct_score_norm'] -
                               pre['correct_score_norm'], 4)
            mc_changed = pre['predicted_idx'] != post['predicted_idx']
            scvr_flag  = (rw_post > rw_pre) and not mc_changed

            row = {
                'truthfulqa_id':          tqa_id,
                'label':                  item['label'],
                'category':               category,
                'method':                 method_name,
                'num_options':            num_options,
                'rewrite_pre':            round(rw_pre, 4),
                'rewrite_post':           round(rw_post, 4),
                'rewrite_delta':          round(rw_post - rw_pre, 4),
                'rewrite_improved':       rw_post > rw_pre,
                'mc_score_pre':           round(pre['correct_score_norm'], 4),
                'mc_score_post':          round(post['correct_score_norm'], 4),
                'mc_delta':               mc_delta,
                'pre_predicted_correct':  pre['predicted_correct'],
                'post_predicted_correct': post['predicted_correct'],
                'mc_changed':             mc_changed,
                'scvr_flag':              scvr_flag,
                'error':                  ''
            }
            rows.append(row)
            done_n += 1
            print(f"[{done_n:03d}/{total}] {method_name} | "
                  f"{item['label'][:32]:<32} | "
                  f"rw:{rw_pre:.2f}->{rw_post:.2f} | "
                  f"mc_chg:{mc_changed} | scvr:{scvr_flag}")

        except Exception as e:
            print(f"  ERR {method_name} {item['label'][:28]}: {str(e)[:70]}")
            rows.append({
                'truthfulqa_id': tqa_id, 'label': item['label'],
                'category': category, 'method': method_name,
                'num_options': num_options, 'error': str(e)[:200]
            })
            done_n += 1

        pd.DataFrame(rows).to_csv(SAVE_PATH, index=False)

# ── Step 10: Final summary ─────────────────────────────────────
del model
torch.cuda.empty_cache(); gc.collect()

df    = pd.DataFrame(rows)
valid = df[df['error'].isna() | (df['error'] == '')]

print(f"\n{'='*60}")
print("TRUTHFULQA CROSS-DATASET VALIDATION — FINAL RESULTS")
print(f"{'='*60}")

for method in ['ROME', 'WISE']:
    m = valid[valid['method'] == method]
    if len(m) == 0: continue
    print(f"\n{method} ({len(m)} valid / {len(selected)} items):")
    print(f"  Rewrite improved:   {m['rewrite_improved'].sum()}/{len(m)} ({100*m['rewrite_improved'].mean():.1f}%)")
    print(f"  MC changed:         {m['mc_changed'].sum()}/{len(m)} ({100*m['mc_changed'].mean():.1f}%)")
    print(f"  SCVR flag:          {m['scvr_flag'].sum()}/{len(m)}")
    print(f"  Mean mc_delta:      {m['mc_delta'].mean():.4f}")

print(f"\n{'='*60}")
print("SCVR BY CATEGORY")
print(f"{'='*60}")
cat_summary = (valid.groupby(['category','method'])
               .agg(n=('label','count'),
                    rw_improved=('rewrite_improved','sum'),
                    mc_changed=('mc_changed','sum'),
                    scvr=('scvr_flag','sum'))
               .reset_index())
print(cat_summary.to_string(index=False))
print(f"\nSaved: {SAVE_PATH}")

import IPython.display as ipd
sr = 22050
t  = np.linspace(0, 0.5, int(sr*0.5))
ipd.display(ipd.Audio(0.3*np.sin(2*np.pi*880*t), rate=sr, autoplay=True))
