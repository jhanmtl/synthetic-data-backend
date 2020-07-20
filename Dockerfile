FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV MAX_WORKERS=2

COPY ./app /app

