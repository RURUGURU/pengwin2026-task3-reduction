FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/app/requirements.txt
COPY inference /opt/app/inference
COPY baseline  /opt/app/baseline
ENV PYTHONPATH=/opt/app:/opt/app/inference:/opt/app/baseline HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/.cache
RUN groupadd -r user && useradd --no-log-init -r -g user user
USER user:user
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]
