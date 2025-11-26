FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y ca-certificates gnupg wget \
    && apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libgtk-3-0 \
    # install other system deps as per Playwright docs...
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
# ensure Playwright browsers are installed
RUN playwright install --with-deps

COPY . .
EXPOSE 8000
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
