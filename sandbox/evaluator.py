import os
import json
import subprocess
import traceback
from typing import Dict, List, Any
import concurrent.futures as cfuts
from tqdm import tqdm
import re

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
    """Load and validate dataset from JSON file"""
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
    """Load answers from JSONL file with error handling"""
    answers_path = os.path.join(CONFIG["base_dir"], CONFIG["answers_file"])
    answers = {}
    
    with open(answers_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                problem_id = entry['metadata']['problem_id']
                answers[problem_id] = entry['code']
            except Exception as e:
                print(f"Error in line {line_num}: {str(e)}")
    
    return answers

def preprocess_code(raw_code: str) -> str:
    code = raw_code.strip()
    if "```python" in code:
        # Extract code between the first occurrence of ```python and the next ```
        code = code.split("```python", 1)[1].split("```", 1)[0]
    return code.strip()

def build_test_program(problem: Dict, solution: str) -> str:
    """Construct executable test code with validation"""
    try:
        context = problem["code_context"]
        return context.replace('[insert]', solution)
    except KeyError as e:
        raise ValueError(f"Missing key in problem: {str(e)}")

def execute_in_docker(code: str) -> Dict[str, Any]:
    """Run code in Docker container with detailed diagnostics"""
    try:
        cmd = [
            'docker', 'run', '--rm', '-i',
            '--network=none',
            '--memory=512m',
            '--cpus=1',
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
    except subprocess.TimeoutExpired as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Timeout after {CONFIG['timeout']} seconds",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "exit_code": -2
        }

def evaluate_problems(dataset: List[Dict], answers: Dict[int, str]) -> Dict:
    """Main evaluation process with detailed error tracking"""
    detailed_results = []
    stats = {
        "total_problems": 0,
        "passed_problems": 0,
        "total_test_cases": 0,
        "passed_test_cases": 0
    }

    with cfuts.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        for problem in dataset:
            problem_id = problem["metadata"]["problem_id"]
            stats["total_problems"] += 1
            stats["total_test_cases"] += problem["metadata"]["test_case_cnt"]
            
            try:
                raw_code = answers.get(problem_id, "")
                clean_code = preprocess_code(raw_code)
                test_program = build_test_program(problem, clean_code)
                futures.append((problem, executor.submit(execute_in_docker, test_program)))
            except Exception as e:
                detailed_results.append({
                    "problem_id": problem_id,
                    "status": "preprocessing_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "solution_code": raw_code[:500],
                    "trace": traceback.format_exc()
                })

        for problem, future in tqdm(futures, total=len(futures)):
            result = future.result()
            problem_id = problem["metadata"]["problem_id"]
            test_case_cnt = problem["metadata"]["test_case_cnt"]
            
            entry = {
                "problem_id": problem_id,
                "library": problem["metadata"]["library"],
                "test_case_count": test_case_cnt,
                "solution_code": answers.get(problem_id, "")[:500],
                "docker_result": result
            }
            
            if result["status"] == "success":
                stats["passed_problems"] += 1
                stats["passed_test_cases"] += test_case_cnt
                entry["status"] = "passed"
            else:
                entry.update({
                    "status": "failed",
                    "error_type": "execution",
                    "error_message": result["stderr"] or result["stdout"]
                })
            
            detailed_results.append(entry)

    return {
        "detailed_results": detailed_results,
        "statistics": stats
    }

def save_reports(results: Dict):
    """Save both detailed and summary reports"""
    os.makedirs(os.path.join(CONFIG["base_dir"], CONFIG["results_dir"]), exist_ok=True)
    
    # Save detailed error report
    detailed_path = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"], 
                               f"{CONFIG['model_name']}_detailed_report.json")
    with open(detailed_path, 'w') as f:
        json.dump(results["detailed_results"], f, indent=2)
    
    # Generate and save summary report
    stats = results["statistics"]
    problem_acc = (stats["passed_problems"] / stats["total_problems"]) * 100 if stats["total_problems"] > 0 else 0
    test_case_acc = (stats["passed_test_cases"] / stats["total_test_cases"]) * 100 if stats["total_test_cases"] > 0 else 0
    
    summary = f"""Evaluation Report for {CONFIG['model_name']}
Total Problems: {stats['total_problems']}
Passed Problems: {stats['passed_problems']}
Problem Accuracy: {problem_acc:.1f}%
Test Cases Passed: {test_case_acc:.1f}%"""
    
    summary_path = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"], 
                              f"{CONFIG['model_name']}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Reports saved to:\n- {detailed_path}\n- {summary_path}")
    print("\n" + summary)

if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset = load_dataset()
        
        print("Loading answers...")
        answers = load_answers()
        
        print("Starting evaluation...")
        results = evaluate_problems(dataset, answers)
        
        print("Saving reports...")
        save_reports(results)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(traceback.format_exc())