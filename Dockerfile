FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 MALLOC_MMAP_THRESHOLD_=131072 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
