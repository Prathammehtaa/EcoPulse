FROM python:3.11-slim

WORKDIR /app

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