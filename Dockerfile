FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Create logs directory to prevent startup errors
RUN mkdir -p logs
EXPOSE 5002
CMD ["python", "app.py"]