# Image to train the model and generate predictions for the credit-risk project.
FROM python:3.12-slim

# libgomp1 provides the OpenMP runtime that LightGBM and scikit-learn need.
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the image.
COPY src/ ./src/

# data/ and models/ are NOT copied in (see .dockerignore); mount them at runtime:
#   docker build -t credit-risk .
#   docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" credit-risk
# By default the container trains the model. Override to predict:
#   docker run --rm -v ... credit-risk python src/predict.py
CMD ["python", "src/train.py"]