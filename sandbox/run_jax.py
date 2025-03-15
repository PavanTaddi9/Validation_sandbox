import sys
import json
import resource
from typing import Dict, Any

# Security constraints
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))

def safe_execute(code: str) -> Dict[str, Any]:
    """Execute preprocessed code in isolated environment"""
    result = {"status": "unknown", "output": ""}
    try:
        # Restricted globals for JAX
        restricted_globals = {
            '__builtins__': {
                'range': range, 'list': list, 'tuple': tuple,
                'dict': dict, 'float': float, 'int': int,
                'bool': bool, 'str': str, 'len': len, 'print': print
            },
            'jax': __import__('jax'),
            'jax.numpy': __import__('jax.numpy'),
            'np': __import__('jax.numpy')
        }
        
        exec(code, restricted_globals)
        result["status"] = "success"
    except Exception as e:
        result.update({
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
    return result

if __name__ == "__main__":
    input_code = sys.stdin.read()
    print(json.dumps(safe_execute(input_code)))