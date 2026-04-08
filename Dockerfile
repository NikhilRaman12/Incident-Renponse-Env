FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

RUN useradd -m -u 1000 user
USER user

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
