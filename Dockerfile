FROM python:3.13-slim

WORKDIR /app

# Install system dependencies that might be needed by ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set necessary environment variables
ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=True
ENV OMP_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "src.serving.serve:app", "--host", "0.0.0.0", "--port", "8000"]
