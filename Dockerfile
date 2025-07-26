FROM python:slim@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419
WORKDIR /app
RUN apt update && apt install -y ffmpeg libsm6 libxext6 && apt clean && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt ./requirements.txt
RUN pip install --break-system-packages -r requirements.txt
COPY ./*.py ./
ENTRYPOINT ["python3", "/app/splitter.py"]
WORKDIR /in
WORKDIR /out