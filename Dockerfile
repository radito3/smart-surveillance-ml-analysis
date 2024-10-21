FROM python:3.12-bookworm

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
# TODO: generate proto compiled files

# disable buffering for stdout and stderr to avoid log output loss on app crash
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
CMD ["app.py"]
