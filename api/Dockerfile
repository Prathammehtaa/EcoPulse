FROM python:3.13-slim

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY Model_Pipeline/src/ ./Model_Pipeline/src/
COPY api/ ./api/

ENV PYTHONUNBUFFERED=1
ENV GCS_BUCKET=ecopulse-lakehouse
ENV MODELS_GCS_PREFIX=models
ENV DATA_GCS_PREFIX=processed

COPY api/startup.sh ./startup.sh
RUN chmod +x ./startup.sh

EXPOSE 8000

ENTRYPOINT ["./startup.sh"]