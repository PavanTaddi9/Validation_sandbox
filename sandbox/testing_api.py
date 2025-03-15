import json
import os
import sys
import gzip
import time
import traceback
import concurrent.futures as cfuts

from tqdm import tqdm
from openai import OpenAI

# Check API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def call_openai_api(system_prompt, prompt, temperature, n, model, max_tokens, stop) -> list[str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    retries = 0
    while True:
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                stop=stop
            )
            return [choice.message.content for choice in result.choices]
        except Exception as e:
            retries += 1
            print(f"⚠️ API call failed (retry {retries}): {e}")
            traceback.print_exc()
            time.sleep(10)

def run_openai(model: str, temperature: float):
    def run_func(prompt: str, id: int):
        default_prompt = (
            "Write a short code following the given format and indentation. "
            "Place the executable code between <code> and </code> tags, without any other non-executable things."
        )
        if "gpt-4o" in model:
            default_prompt = "Only provide the code completion needed. Don't repeat the context code."

        code = call_openai_api(
            default_prompt,
            prompt,
            temperature=temperature,
            n=1,
            model=model,
            max_tokens=1024,
            stop=["</code>", "# SOLUTION END"],
        )

        return {
            'id': id,
            'code': code[0],
            'metadata': ds1000[id]['metadata']
        }

    ds1000_responses = []
    with cfuts.ThreadPoolExecutor(max_workers=16) as executor:
        futs = []
        for i, p in enumerate(ds1000):
            try:
                id = int(p['metadata']['problem_id'])
                prompt = p['prompt']
                futs.append(executor.submit(run_func, prompt, id))
            except Exception as e:
                print(f"❌ Error submitting job for item {i}: {e}")
                traceback.print_exc()

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            try:
                result = f.result()
                ds1000_responses.append(result)
            except Exception as e:
                print(f"❌ Error getting result: {e}")
                traceback.print_exc()

    ds1000_responses.sort(key=lambda x: int(x['id']))
    output_file = f'data/{model}-answers.jsonl'
    os.makedirs("data", exist_ok=True)

    with open(output_file, 'w') as f:
        for r in ds1000_responses:
            f.write(json.dumps(r) + '\n')

    print(f"✅ Results written to {output_file}")

def load_ds1000(file_path="jax_M0_questions_updated.json"):
    if not os.path.exists(file_path):
        print(f"❌ Dataset not found at {file_path}")
        sys.exit(1)
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    print("🔍 Loading dataset...")
    ds1000 = load_ds1000()
    print(f"📦 Loaded {len(ds1000)} problems.")
    run_openai('gpt-4o-mini', 0)