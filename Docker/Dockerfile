FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1 \
        libglew-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
ENV PYTHONPATH=/app
ENV MUJOCO_GL=osmesa

ENTRYPOINT ["python", "Docker/app.py"]
CMD ["smoke"]
