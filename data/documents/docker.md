# Docker Containers

Docker packages an application and its dependencies into a container.

A container provides an isolated runtime environment for an application.

## Dockerfile

A Dockerfile describes how to build a container image.

A simple Python application might use:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
