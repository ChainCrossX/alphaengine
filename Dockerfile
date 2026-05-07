FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn flask

COPY . .

# Sanity check at build time: fail the image build, not the runtime, if the
# package layout is wrong. Saves an hour of "why isn't gunicorn finding it".
RUN test -f /app/alphaengine/__init__.py || (\
    echo "ERROR: alphaengine/ package missing from build context."; \
    echo "       Confirm your repo has alphaengine/__init__.py at the root."; \
    ls -la /app; exit 1)

ENV PORT=8000 PYTHONPATH=/app
EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "alphaengine.web.app:app"]
