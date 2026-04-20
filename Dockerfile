FROM python:3.11-slim

RUN python3 -m venv /app/venv

COPY requirements.txt /app/requirements.txt

RUN /app/venv/bin/pip3 install -r /app/requirements.txt

COPY . /app

WORKDIR /app

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/venv/bin/python3", "/app/main.py"]
