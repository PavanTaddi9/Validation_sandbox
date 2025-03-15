import os
import json
import subprocess
from typing import List, Dict, Any
import concurrent.futures as cfuts
from tqdm import tqdm

def preprocess(code: List | str) -> str:
    """Clean raw model output before execution"""
    if isinstance(code, list):
        code = code[0]
    code = code.split('</code>')[0]
    code = code.replace('```python', '')
    code = code.split('```')[0]
    code = code.split('\nEND SOLUTION')[0]
    return code.replace('<code>', '')

def postprocess_results(results: List[Dict]) -> str:
    """Analyze execution results and generate report"""
    total_score = sum(res['score'] for res in results)
    lib_scores = {}
    for res in results:
        lib = res['library']
        lib_scores[lib] = lib_scores.get(lib, 0) + res['score']
    
    summary = f"Total Score: {total_score}/1000\n"
    for lib, score in lib_scores.items():
        count = len([x for x in results if x['library'] == lib])
        summary += f"{lib}: {score}/{count}\n"
    return summary

def execute_in_sandbox(code: str, timeout: int) -> Dict[str, Any]:
    """Run preprocessed code in Docker sandbox"""
    try:
        cmd = [
            'docker', 'run', '--rm', '-i',
            '--network=none',
            '--memory=256m',
            'jax-sandbox'
        ]
        
        result = subprocess.run(
            cmd,
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        
        return json.loads(result.stdout.decode())
    
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"status": "error", "error_message": str(e)}

def evaluate_submissions(answers: List[str]) -> str:
    """Main evaluation pipeline"""
    processed_results = []
    
    with cfuts.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx, raw_code in enumerate(answers):
            # Preprocessing stage
            clean_code = preprocess(raw_code)
            futures.append(executor.submit(
                execute_in_sandbox, 
                clean_code,
                120
            ))
        
        # Process results with progress bar
        for future in tqdm(cfuts.as_completed(futures), total=len(futures)):
            result = future.result()
            # Convert sandbox result to scoring format
            processed_results.append({
                'score': 1 if result['status'] == 'success' else 0,
                'reason': result.get('error_message', ''),
                'library': 'JAX'  # Assuming all are JAX problems
            })
    
    return postprocess_results(processed_results)