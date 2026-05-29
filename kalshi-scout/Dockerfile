# Multi-stage build for kalshi-scout.
# - Stage 1 installs the package + deps into a virtualenv.
# - Stage 2 copies the venv into a slim runtime image with /data as the
#   default snapshot-store mount point.

FROM python:3.11-slim AS build
WORKDIR /app

# System deps for httpx + sqlite (sqlite3 ships with Python, no apt needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install into a virtualenv we can copy intact.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .


FROM python:3.11-slim
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Snapshot store + alerts log persist under /data; mount a volume here.
VOLUME ["/data"]

# Default: run the daemon every 5 minutes, alert via stdout + jsonl.
# Override via `docker run ... kalshi-scout ...` or compose.
ENTRYPOINT ["kalshi-scout"]
CMD ["serve", \
     "--store", "/data/scout.db", \
     "--interval", "300", \
     "--min-grade", "C", \
     "--notify", "stdout", \
     "--notify", "jsonl:/data/alerts.jsonl", \
     "--notify-min-grade", "A"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('/data/scout.db', timeout=2).execute('SELECT 1').fetchone()" || exit 1
