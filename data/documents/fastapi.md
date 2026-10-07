# FastAPI

FastAPI is a Python framework for building APIs.

FastAPI uses Python type hints to define request and response data.

## Basic API

A simple FastAPI application can be written as:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}