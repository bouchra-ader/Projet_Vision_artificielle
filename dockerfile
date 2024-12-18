# Use a slim version of Python 3.9
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory to the working directory in the container
COPY . .

# Install gunicorn for production
CMD ["gunicorn", "app:app"]
