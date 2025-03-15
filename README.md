# DS-1000 JAX Code Evaluation System

A pipeline for evaluating code submissions against **JAX-converted questions from the DS-1000 benchmark**, originally containing NumPy problems that have been adapted to JAX syntax and semantics.

## 📋 Overview

This system evaluates LLM-generated code solutions for JAX programming problems derived from the DS-1000 (modified)benchmark. Key components:

- **Dataset Origin**: Contains questions converted from NumPy to JAX from the original DS-1000 benchmark
- **Secure Sandboxing**: Docker containers for isolated execution
- **JAX-Specific Adaptation**: Solutions evaluated against JAX-specific implementations
- **Detailed Analytics**: Problem-level and perturbation-type insights

## 🚀 Features

1. **Dockerized Execution Environment**
   - Resource constraints (RAM/CPU)
   - Network isolation
   - Automatic cleanup

2. **Preprocessing**
   - Markdown stripping
   - Code sanitization
   - Indentation correction

3. **Multi-Level Reporting**
   - Aggregate accuracy statistics
   - Perturbation-type breakdowns
   - Raw execution logs
   - Error classification

4. **Performance Optimization**
   - Parallel worker threads
   - Progress tracking
   - Timeout handling

## 🛠️ Setup

### Requirements
- Docker Engine ≥20.10
- Python 3.8+
- 4GB+ free memory

### Installation
```bash
git clone https://github.com/PavanTaddi9/Validation_sandbox.git
cd jax-code-evaluator

# Build Docker image
docker build -t jax-sandbox -f docker/Dockerfile .

# Create data directories
mkdir -p data results


├── data/
│   ├── jax_M0_questions_updated.json    # Problem dataset
│   └── gpt-4o-mini-answers.jsonl        # Model submissions
├── docker/
│   └── Dockerfile                       # Sandbox configuration
├── src/
│   └── evaluator.py                     # Main evaluation logic
└── results/                             # Generated reports