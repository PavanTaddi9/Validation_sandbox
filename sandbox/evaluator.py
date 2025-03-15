import os
import json
import subprocess
from typing import Dict, List, Any
import concurrent.futures as cfuts
from tqdm import tqdm

# ========================
# CONFIGURATION
# ========================
CONFIG = {
    "model_name": "gpt-4o-mini",
    "base_dir": "/Users/pavankumartaddi/Desktop/TESTING/sandbox",
    "dataset_file": "data/jax_M0_questions_updated.json",
    "answers_file": "data/gpt-4o-mini-answers.jsonl",
    "results_dir": "results",
    "docker_image": "jax-sandbox",
    "timeout": 120,
    "max_workers": 8
}
# ========================

def load_dataset() -> List[Dict]:
    """Load dataset from JSON file"""
    dataset_path = os.path.join(CONFIG["base_dir"], CONFIG["dataset_file"])
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset should be a list of problems")
    
    required_keys = {"prompt", "reference_code", "metadata", "code_context"}
    for idx, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Problem {idx} missing keys: {missing}")
    
    return data

def load_answers() -> Dict[int, str]:
    """Load answers from JSONL file"""
    answers_path = os.path.join(CONFIG["base_dir"], CONFIG["answers_file"])
    answers = {}
    
    with open(answers_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                problem_id = entry['metadata']['problem_id']
                answers[problem_id] = entry['code']
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Skipping invalid answer entry: {str(e)}")
    
    return answers

def preprocess_code(raw_code: str) -> str:
    """Clean code from markdown blocks"""
    code = raw_code.strip()
    if '```python' in code:
        code = code.split('```python')[1].split('```')[0]
    return code.strip()

def build_test_program(problem: Dict, solution: str) -> str:
    """Construct executable test code"""
    context = problem["code_context"]
    return context.replace('[insert]', solution)

def execute_in_docker(code: str) -> Dict[str, Any]:
    """Run code in Docker container"""
    try:
        cmd = [
            'docker', 'run', '--rm', '-i',
            '--network=none',
            '--memory=512m',
            CONFIG["docker_image"]
        ]
        
        result = subprocess.run(
            cmd,
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=CONFIG["timeout"]
        )
        
        return {"status": "success" if result.returncode == 0 else "error",
                "output": result.stdout.decode(),
                "error": result.stderr.decode()}
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "Timeout"}

def evaluate_problems(dataset: List[Dict], answers: Dict[int, str]) -> Dict[str, float]:
    """Main evaluation process"""
    results = []
    total_tests = 0
    passed_tests = 0
    
    with cfuts.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        for problem in dataset:
            problem_id = problem["metadata"]["problem_id"]
            raw_code = answers.get(problem_id, "")
            
            try:
                clean_code = preprocess_code(raw_code)
                test_program = build_test_program(problem, clean_code)
                futures.append(executor.submit(execute_in_docker, test_program))
            except Exception as e:
                futures.append(executor.submit(
                    lambda: {"status": "error", "reason": str(e)}
                ))

        for idx, future in enumerate(tqdm(futures, total=len(futures))):
            problem = dataset[idx]
            test_case_cnt = problem["metadata"]["test_case_cnt"]
            result = future.result()
            
            passed = result["status"] == "success"
            total_tests += test_case_cnt
            passed_tests += test_case_cnt if passed else 0
            results.append(passed)

    return {
        "total_problems": len(dataset),
        "passed_problems": sum(results),
        "problem_accuracy": sum(results)/len(dataset),
        "test_case_accuracy": passed_tests/max(total_tests, 1)
    }

def save_results(results: Dict[str, float]):
    """Save evaluation results"""
    report = f"""Evaluation Report for {CONFIG['model_name']}
Total Problems: {results['total_problems']}
Passed Problems: {results['passed_problems']}
Problem Accuracy: {results['problem_accuracy']*100:.1f}%
Test Cases Passed: {results['test_case_accuracy']*100:.1f}%"""
    
    results_dir = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"{CONFIG['model_name']}_results.txt"), 'w') as f:
        f.write(report)
    
    print(report)

if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset = load_dataset()
        
        print("Loading answers...")
        answers = load_answers()
        
        print("Evaluating submissions...")
        results = evaluate_problems(dataset, answers)
        
        print("\nSaving results...")
        save_results(results)
        
    except Exception as e:
        print(f"\nError: {str(e)}")