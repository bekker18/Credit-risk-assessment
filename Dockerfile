FROM python:3.12-slim

WORKDIR /app

# LightGBM needs OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Make src imports work consistently
ENV PYTHONPATH=/app/src

# Default command: generate submission using the saved model
CMD ["python", "src/predict.py"]