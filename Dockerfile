# TicketRouterEnv — OpenEnv-compatible image (FastAPI + optional baseline inference)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Hugging Face Space / OpenEnv: simulation mode exposes POST /reset, /step, GET /state, /health
# Run baseline locally with: docker run -e TASK=ticket_router_easy -e HF_TOKEN=... this-image python inference.py
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
