FROM python:3.12-bookworm

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

RUN python setup.py build_ext --inplace

ENTRYPOINT ["python"]
CMD ["app.py"]
