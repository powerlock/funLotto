FROM python:3.9

WORKDIR .

COPY . .

RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
ENV FLASK_APP serve_completed.py

CMD ["uvicorn","serve_completed:app","--host","0.0.0.0","--port","8000"]