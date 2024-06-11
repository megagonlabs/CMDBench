FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY docker/services/demo .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "🏠_Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless", "1", "--theme.base", "dark"]
