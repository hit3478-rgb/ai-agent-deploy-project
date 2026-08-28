FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 최소 설치 (scikit-learn 빌드에 필요할 수 있는 것들)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 실제 실행 커맨드는 docker-compose.yml에서 서비스별로 지정
# (backend: uvicorn, frontend: streamlit — 같은 이미지, 다른 커맨드)
