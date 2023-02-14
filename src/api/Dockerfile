FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app

COPY . /app/

EXPOSE 8080

CMD ["python", "./app.py"]