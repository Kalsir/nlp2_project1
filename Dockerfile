FROM continuumio/miniconda3

WORKDIR /app

COPY . /tmp

RUN pip install -r /tmp/requirements.txt

CMD ["python", "main.py"]
