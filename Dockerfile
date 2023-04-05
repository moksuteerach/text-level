# Set the base image to use
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port your Flask application will listen on
EXPOSE 5000

# Start the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]