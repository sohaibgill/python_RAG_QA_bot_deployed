FROM python:3.12.1

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install python3-pip git 
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]