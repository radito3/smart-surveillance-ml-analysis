FROM python:3.12-bookworm

WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["./start.sh"]
