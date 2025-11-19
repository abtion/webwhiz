###################
# BUILD FOR PRODUCTION
###################

FROM python:3.11-slim-trixie

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update && \
    apt-get install -y -q \
        build-essential \
        curl \
        libpq-dev \
        libxml2-dev \
        libxslt-dev \
        openssl \
        python3-dev \
        && \
    apt-get clean

# Copy project files
COPY . /app
COPY .env.docker /app/.env

# Set working directory
WORKDIR /app

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start Celery worker
CMD celery -A worker worker -l info --concurrency 2
