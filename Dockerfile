FROM python:3.12-bookworm

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]
