FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt || true

COPY inference.py .
CMD ["python", "inference.py"]
