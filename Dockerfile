FROM python:3.10

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

ENTRYPOINT ["streamlit", "run", "sod/app.py"]
