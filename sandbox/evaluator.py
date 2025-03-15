import os
import json
import subprocess
import traceback
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
        return json.load(f)

def load_answers() -> Dict[int, str]:
    """Load answers from JSONL file"""
    answers = {}
    answers_path = os.path.join(CONFIG["base_dir"], CONFIG["answers_file"])
    
    with open(answers_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                problem_id = entry['metadata']['problem_id']
                answers[problem_id] = entry['code']
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Skipping invalid answer entry: {str(e)}")
    
    return answers

def sanitize_code(code: str) -> str:
    """Clean code while preserving existing indentation"""
    code = code.replace('```python', '').replace('```', '')
    code = code.replace('BEGIN SOLUTION', '').replace('END SOLUTION', '')
    return '\n'.join([line.rstrip() for line in code.split('\n') if line.strip()])

def get_indentation_level(context: str) -> str:
    """Detect indentation at the [insert] point"""
    for line in context.split('\n'):
        if '[insert]' in line:
            return line.split('[insert]')[0]
    return ''

def build_test_program(problem: Dict, solution: str) -> str:
    """Insert solution with proper context indentation"""
    context = problem["code_context"]
    indent = get_indentation_level(context)
    
    # Indent solution lines according to context
    indented_solution = '\n'.join([f"{indent}{line}" for line in solution.split('\n')])
    
    return context.replace('[insert]', indented_solution)

def execute_in_docker(code: str) -> Dict[str, Any]:
    """Execute code in Docker container with diagnostics"""
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
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout.decode(),
            "stderr": result.stderr.decode(),
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Timeout after {CONFIG['timeout']}s",
            "exit_code": -1
        }

def evaluate_problems(dataset: List[Dict], answers: Dict[int, str]) -> List[Dict]:
    """Main evaluation with detailed error tracking"""
    results = []
    
    with cfuts.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        for problem in dataset:
            problem_id = problem["metadata"]["problem_id"]
            raw_code = answers.get(problem_id, "")
            
            try:
                clean_code = sanitize_code(raw_code)
                test_program = build_test_program(problem, clean_code)
                futures.append((problem, executor.submit(execute_in_docker, test_program)))
            except Exception as e:
                results.append({
                    "problem_id": problem_id,
                    "error": f"Preparation failed: {str(e)}",
                    "trace": traceback.format_exc()
                })

        for problem, future in tqdm(futures, total=len(futures)):
            problem_id = problem["metadata"]["problem_id"]
            result = future.result()
            
            error_info = {
                "problem_id": problem_id,
                "test_case_count": problem["metadata"]["test_case_cnt"],
                "solution_code": answers.get(problem_id, "")[:500],
                **result
            }
            
            results.append(error_info)
    
    return results

def save_detailed_report(results: List[Dict]):
    """Generate comprehensive error report"""
    report_path = os.path.join(
        CONFIG["base_dir"],
        CONFIG["results_dir"],
        f"{CONFIG['model_name']}_detailed_report.json"
    )
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    os.makedirs(os.path.join(CONFIG["base_dir"], CONFIG["results_dir"]), exist_ok=True)
    
    try:
        print("Loading dataset...")
        dataset = load_dataset()
        
        print("Loading answers...")
        answers = load_answers()
        
        print("Starting evaluation...")
        results = evaluate_problems(dataset, answers)
        
        print("Saving report...")
        save_detailed_report(results)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(traceback.format_exc())