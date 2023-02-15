FROM python:3.9-slim

COPY requirements.txt \
     setup.py \
     ./

RUN pip install -r requirements.txt &&\
    pip install -e .

COPY . ./

EXPOSE 8080

CMD ["python", "src/api/app.py"]