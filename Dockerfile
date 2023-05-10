FROM python:3.9-buster
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]