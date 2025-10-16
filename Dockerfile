# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only essential files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app/ app/
COPY models/ models/

# Expose Flask port
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app/mlb_predict.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask app
CMD ["flask", "run"]
