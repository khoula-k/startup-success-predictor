FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests pandas

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]