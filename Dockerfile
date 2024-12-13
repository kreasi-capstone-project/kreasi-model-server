# Use the official Python image as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE

# Set the working directory in the container
WORKDIR /app

# Copy the application code and model files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "main:app"]
