FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 4242

COPY docker/services/demo/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY docker/services/demo .

HEALTHCHECK CMD curl --fail http://localhost:4242/_stcore/health

ENTRYPOINT ["streamlit", "run", "🏠_Home.py", "--server.port=4242", "--server.address=0.0.0.0", "--server.headless", "1", "--theme.base", "dark"]
