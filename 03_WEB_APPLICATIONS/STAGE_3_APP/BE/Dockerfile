# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend app
COPY app.py .

# Expose port
EXPOSE 5000

# Run Flask
CMD ["python", "app.py"]