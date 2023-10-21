FROM python:3.10.7

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
COPY . /src/
CMD [ "python3", "app/main.py" ]