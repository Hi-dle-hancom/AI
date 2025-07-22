# translator-service/Dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch 인덱스 URL 설정
RUN pip config set global.index-url https://pypi.org/simple
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 번역 서비스는 8003번 포트 사용
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]