FROM python:3.12-bookworm

WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt

# disable buffering for stdout and stderr to avoid log output loss on app crash
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["./start.sh"]
