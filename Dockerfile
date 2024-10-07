# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg libmagic1

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV CLOUD_TASKS_QUEUE public-chat
ENV CLOUD_TASKS_QUEUE_REGION us-central1
ENV GOOGLE_CLOUD_PROJECT cvs-innovation-demo
ENV GOOGLE_CLOUD_PROJECT_NUMBER 179280619779
ENV GOOGLE_CLOUD_REGION us-central1
ENV GOOGLE_CLOUD_BUCKET cvs-innovation-demo.appspot.com

# Run main.py when the container launches
CMD ["python", "main.py"]
