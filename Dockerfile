FROM continuumio/miniconda3

WORKDIR /app/src

COPY . /tmp

RUN pip install -r /tmp/requirements.txt

CMD ["python", "main.py"]
