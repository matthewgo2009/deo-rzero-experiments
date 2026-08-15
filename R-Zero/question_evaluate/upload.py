import json
import huggingface_hub
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
import json
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
print(STORAGE_PATH)
with open('tokens.json', 'r') as f:
    token = json.load(f)['huggingface']
login(token=token)
parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
args = parser.parse_args()

datas= []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue


for i in range(8):
    try:
        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')

#count the number  of score between 0.2 and 0.8
if not args.repo_name == "":
    filtered_datas = [{'problem':data['question'],'answer':data['answer'],'score':data['score']} for data in datas if data['score'] >= args.min_score and data['score'] <= args.max_score and data['answer'] != '' and data['answer']!= 'None']
    print(len(filtered_datas))

    # --- RZ_LABEL=claude: replace the solver majority-vote answer with Claude's solve ---
    # (same labeler as DEO canonical_claude_label; questions Claude can't box are dropped).
    # RZ_LABEL_CAP: random-subsample the filtered set first (verl consumes only 1280
    # prompts/iter anyway; capping bounds API cost).
    if os.getenv("RZ_LABEL", "") == "claude" and filtered_datas:
        import random, re, time, urllib.request, urllib.error
        from concurrent.futures import ThreadPoolExecutor

        cap = int(os.getenv("RZ_LABEL_CAP", "0"))
        if cap and len(filtered_datas) > cap:
            random.seed(0)
            filtered_datas = random.sample(filtered_datas, cap)
            print(f"[claude-label] capped filtered set to {cap}")

        LABEL_MODEL = os.getenv("RZ_LABEL_MODEL", "claude-sonnet-4-5-20250929")
        AKEY = json.load(open('tokens.json'))['anthropic']
        SYSTEM = ("You are an expert mathematician. Solve the problem and give ONLY the final "
                  "answer inside \\boxed{...} at the end.")
        _BOX = re.compile(r"\\boxed\{")

        def _last_boxed(text):
            starts = [m.end() for m in _BOX.finditer(text or "")]
            if not starts:
                return None
            pos, depth, out = starts[-1], 1, []
            while pos < len(text) and depth > 0:
                c = text[pos]
                if c == "{": depth += 1; out.append(c)
                elif c == "}":
                    depth -= 1
                    if depth == 0: break
                    out.append(c)
                else: out.append(c)
                pos += 1
            return ("".join(out).strip() or None) if depth == 0 else None

        def _solve(q, retries=6):
            msgs = [{"role": "user", "content": q}]
            for a in range(retries):
                try:
                    body = json.dumps({"model": LABEL_MODEL, "max_tokens": 16384,
                                       "system": SYSTEM, "messages": msgs}).encode()
                    req = urllib.request.Request(
                        "https://api.anthropic.com/v1/messages", data=body,
                        headers={"x-api-key": AKEY, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"})
                    with urllib.request.urlopen(req, timeout=300) as r:
                        d = json.load(r)
                    text = "".join(b.get("text", "") for b in d.get("content", []))
                    boxed = _last_boxed(text)
                    if boxed:
                        return boxed
                    msgs = msgs + [{"role": "assistant", "content": text or "(no answer)"},
                                   {"role": "user", "content": "Output ONLY your final answer as \\boxed{ANSWER}."}]
                except urllib.error.HTTPError as e:
                    time.sleep((5 if e.code in (429, 529) else 2) * (a + 1))
                except Exception:
                    time.sleep(2 * (a + 1))
            return None

        with ThreadPoolExecutor(max_workers=int(os.getenv("RZ_LABEL_WORKERS", "8"))) as ex:
            claude_answers = list(ex.map(lambda d: _solve(d['problem']), filtered_datas))
        kept, agree = [], 0
        for d, ca in zip(filtered_datas, claude_answers):
            if ca is None:
                continue
            if ca.strip() == str(d['answer']).strip():
                agree += 1
            kept.append({'problem': d['problem'], 'answer': ca, 'score': d['score']})
        print(f"[claude-label] relabeled {len(kept)}/{len(filtered_datas)} "
              f"(dropped {len(filtered_datas)-len(kept)} unboxable; "
              f"exact-match agreement with majority vote: {agree}/{len(kept)})")
        filtered_datas = kept
    train_dataset = Dataset.from_list(filtered_datas)
    dataset_dict = {"train": train_dataset}
    config_name = f"{args.experiment_name}"
    dataset = DatasetDict(dataset_dict)
    dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)







