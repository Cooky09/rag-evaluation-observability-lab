# RAG Evaluation & Observability Lab

A local, open-source platform for evaluating RAG/LLM quality while benchmarking inference performance across batching and precision configurations, with n8n used to automate evaluation workflows.

## Overview

This project combines **RAG evaluation, local LLM inference, performance benchmarking, observability, regression testing, and workflow automation** into one reproducible engineering project.

The system is designed to run locally without paid APIs or cloud infrastructure.

### Architecture

```text
                         n8n
                          |
                          v
                     FastAPI
                  Evaluation API
                          |
             +------------+------------+
             |            |            |
             v            v            v
        RAG Pipeline   Evaluation   Benchmark
             |            |            |
         embeddings    RAG/LLM       CPU/GPU
         retrieval     metrics       FP32/FP16
                                      batching
             +------------+------------+
                          |
                          v
                    Observability
                 logs / metrics / GPU
                          |
                          v
                  Reports / quality gate
```

## Key Features

- Local semantic RAG pipeline
- Sentence Transformer embeddings
- Semantic vector retrieval
- Local LLM generation through Ollama
- Document loading and paragraph-aware chunking
- Retrieval evaluation
- Answer similarity evaluation
- Fact coverage evaluation
- Grounded-answer evaluation
- Regression quality gates
- FastAPI REST API
- Prometheus-compatible application metrics
- CPU/GPU capability detection
- Batch inference benchmarking
- FP32 inference benchmarking
- GPU-ready FP16 inference path
- GPU memory measurement when CUDA is available
- Quality-versus-performance reporting
- n8n workflow automation
- Automated pytest test suite
- GitHub Actions CI

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.14 |
| API | FastAPI |
| RAG embeddings | Sentence Transformers |
| Local LLM | Ollama |
| ML framework | PyTorch |
| Retrieval | NumPy semantic similarity |
| Evaluation | Custom evaluation engine |
| Observability | Python logging + Prometheus |
| Automation | n8n |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker / Docker Compose |
| Version control | Git |

## Current Hardware

The project was developed and tested on a Windows laptop with:

- Intel UHD Graphics 620
- No NVIDIA GPU
- CPU-only PyTorch build
- CUDA unavailable

The project therefore does **not** fabricate NVIDIA GPU benchmark results.

Instead, the inference layer detects available hardware and uses CPU when CUDA is unavailable.

The CUDA and FP16 code paths are implemented so that the same project can be tested on an NVIDIA GPU machine later.

## RAG Pipeline

The RAG pipeline consists of:

```text
Documents
    |
    v
Document Loader
    |
    v
Paragraph-aware Chunking
    |
    v
Sentence Transformer Embeddings
    |
    v
Semantic Retrieval
    |
    v
Context Assembly
    |
    v
Local LLM
    |
    v
Answer + Sources
```

The system uses normalized embeddings and cosine-equivalent dot-product similarity for semantic retrieval.

## Evaluation Pipeline

The evaluation dataset contains 10 questions covering:

- Python virtual environments
- Docker
- FastAPI
- CUDA
- Batch inference
- FP16 inference
- GPU benchmarking

Each evaluation case contains a reference answer and expected facts.

The evaluation engine measures:

### Retrieval

- Precision
- Recall
- Expected source matching

### Generation

- Reference-answer similarity
- Fact coverage
- Grounded-answer rate

### Regression

The quality gate currently requires:

```text
Retrieval recall       >= 0.90
Answer similarity      >= 0.25
Fact coverage          >= 0.25
Grounded rate          >= 0.30
```

If any threshold is missed, the evaluation exits with a failure code.

This allows evaluation quality to become an automated CI/workflow decision rather than a manually inspected report.

## Regression Behaviour

The local LLM is nondeterministic, so evaluation results can vary between runs.

For example, one run may produce:

```text
Average retrieval recall: 1.000
Average answer similarity: 0.272
Average fact coverage: 0.300
Grounded rate: 0.400

Regression check: PASS
```

Another run may produce:

```text
Average retrieval recall: 1.000
Average answer similarity: 0.209
Average fact coverage: 0.200
Grounded rate: 0.300

Regression check: FAIL
  - Answer similarity is below the minimum threshold.
  - Fact coverage is below the minimum threshold.
```

This is intentional.

The project demonstrates that the evaluation system can detect quality degradation rather than artificially forcing every run to pass.

## Benchmarking

The benchmark system evaluates inference performance across different batch sizes.

Current CPU benchmark:

| Batch Size | Latency (ms) | Throughput (samples/sec) | Speedup |
|---:|---:|---:|---:|
| 1 | 0.869 | 1,150.61 | 1.00x |
| 2 | 1.103 | 1,814.05 | 1.58x |
| 4 | 1.305 | 3,066.27 | 2.66x |
| 8 | 1.068 | 7,491.90 | 6.51x |
| 16 | 1.418 | 11,282.66 | 9.81x |

The current CPU benchmark shows a substantial throughput improvement as batch size increases.

The highest measured throughput was:

```text
11,282.66 samples/sec
```

at:

```text
FP32
Batch size: 16
CPU
```

The lowest measured latency was:

```text
0.869 ms
```

at:

```text
FP32
Batch size: 1
CPU
```

These results illustrate the difference between **latency optimization** and **throughput optimization**.

## Precision Support

The inference layer supports:

```text
FP32
FP16
```

FP16 is intentionally restricted to supported GPU devices.

On the current CPU-only development machine, CPU FP16 benchmarking is skipped rather than being incorrectly presented as GPU acceleration.

When running on an NVIDIA CUDA environment, the project can evaluate GPU inference and FP16 configurations.

## Observability

The FastAPI application exposes application-level metrics.

### Application metrics

```text
GET /metrics
```

Provides:

- Query count
- Total query duration
- Average query duration
- Total retrieval duration
- Average retrieval duration
- Total generation duration
- Average generation duration

### Prometheus

```text
GET /metrics/prometheus
```

Exposes Prometheus-compatible metrics including:

```text
rag_query_count
rag_average_query_duration_ms
rag_average_retrieval_duration_ms
rag_average_generation_duration_ms
```

### Logging

Application logging records events such as:

```text
RAG query started
Retrieval completed
Generation completed
RAG query completed
Automated evaluation started
Automated evaluation completed
```

## FastAPI API

Start the API with:

```powershell
uvicorn app.main:app --reload
```

### Health

```text
GET /health
```

Example:

```json
{
  "status": "ok",
  "service": "rag-evaluation-observability-lab"
}
```

### RAG Query

```text
POST /query
```

Example request:

```json
{
  "question": "Why should a Python project use a virtual environment?",
  "top_k": 3
}
```

### Application Metrics

```text
GET /metrics
```

### Prometheus Metrics

```text
GET /metrics/prometheus
```

### Automated Evaluation

```text
POST /automation/evaluate
```

The automation endpoint runs the complete evaluation workflow and returns the structured regression result.

Example:

```json
{
  "status": "success",
  "exit_code": 0,
  "regression": {
    "passed": true,
    "failures": []
  }
}
```

When the quality gate fails, the response contains the failed regression conditions.

## n8n Automation

The project includes the exported n8n workflow:

```text
workflows/n8n/evaluation-workflow.json
```

The workflow performs:

```text
Manual Trigger
     |
     v
FastAPI Health Check
     |
     v
Automated Evaluation
     |
     v
Extract Evaluation Metrics
     |
     v
Regression Quality Gate
     |
     +------------------+
     |                  |
    PASS               FAIL
     |                  |
     +---------+--------+
               |
               v
      Evaluation Summary
```

The quality gate uses the structured API response:

```text
$json.regression.passed
```

rather than relying on parsing the process exit code.

This separates:

- **workflow execution status**
- **evaluation quality status**

n8n runs in Docker and communicates with the Windows-hosted FastAPI service through:

```text
http://host.docker.internal:8000
```

## Running the Project

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run tests

```powershell
python -m pytest -q
```

Current test result:

```text
52 passed
2 skipped
```

The skipped tests are hardware-dependent CUDA/FP16 tests.

### 5. Start FastAPI

```powershell
uvicorn app.main:app --reload
```

### 6. Run evaluation

```powershell
python -m scripts.run_evaluation
```

Generated reports:

```text
results/evaluation_report.json
results/quality_performance_report.json
```

These generated JSON reports are intentionally excluded from Git because they are runtime artifacts and can change between runs.

### 7. Run benchmark matrix

```powershell
python -m scripts.benchmark_matrix
```

The benchmark automatically uses CPU and adds CUDA when an NVIDIA-compatible PyTorch environment is available.

## Testing

The project currently contains tests covering:

```text
Document loading
Chunking
Embeddings
Retrieval
RAG generation
Evaluation
Regression gates
FastAPI
Logging
Metrics
Device detection
GPU metrics
Inference configuration
Benchmark model
Precision handling
```

Run the complete suite:

```powershell
python -m pytest -q
```

Expected development-machine result:

```text
52 passed, 2 skipped
```

## CI/CD

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The CI pipeline:

1. Checks out the repository.
2. Installs Python 3.14.
3. Installs the pinned dependencies.
4. Runs the complete pytest suite.

CI runs on pushes and pull requests targeting:

```text
main
master
```

## Project Structure

```text
rag-evaluation-observability-lab/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├──