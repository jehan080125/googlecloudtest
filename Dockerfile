FROM python:3.11-slim

WORKDIR /app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 백엔드 소스코드 복사
COPY . .

# 구글 Cloud Run 환경에 맞춰 포트 설정 (기본값 8080)
ENV PORT 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]