FROM python:3.11-slim

# ... 생략 ...
WORKDIR /app
# 파일을 명시적으로 /app/requirements.txt로 복사
COPY requirements.txt /app/requirements.txt
# 절대 경로를 사용하여 설치
RUN pip install --no-cache-dir -r /app/requirements.txt

# 나머지 백엔드 소스코드 복사
COPY . .

# 구글 Cloud Run 환경에 맞춰 포트 설정 (기본값 8080)
ENV PORT 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
