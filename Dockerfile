FROM python:3.12-bookworm

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# this workaround is done because ultralytics (from which YOLO is from)
# has opencv as a direct dependency and we need opencv-headless instead
RUN pip3 install -r requirements.txt && \
    pip3 install ultralytics --no-deps && \
    pip3 install matplotlib PyYAML pandas py-cpuinfo scipy seaborn ultralytics-thop

ENTRYPOINT ["./start.sh"]
