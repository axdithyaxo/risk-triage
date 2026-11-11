# Build from the repo root: docker build -f app/agent-api/Dockerfile .
# (the API imports the shared pipeline and guidelines from eval/)
FROM python:3.15-slim

# git: requirements.txt installs Lár from a pinned GitHub commit
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eval/lar_agent/agent.py eval/lar_agent/retrieval.py eval/lar_agent/
COPY eval/guidelines eval/guidelines
COPY app/agent-api/main.py app/agent-api/audit_log.py app/agent-api/

# Not root. /data holds the audit log; lar_logs/ is where Lár's flight recorder
# writes (it can contain the raw submission, so compose mounts it as tmpfs).
RUN useradd --system --uid 10001 triage \
    && mkdir -p /data /srv/app/agent-api/lar_logs \
    && chown -R triage /data /srv/app/agent-api
USER triage

WORKDIR /srv/app/agent-api
ENV TRIAGE_AUDIT_LOG_PATH=/data/audit_log.jsonl
VOLUME /data
EXPOSE 8010
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/healthz')"
# Our own middleware logs each request as JSON; uvicorn's plain-text access log would duplicate it.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--no-access-log"]
