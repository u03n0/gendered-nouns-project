# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /code
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the Streamlit example repository
RUN git clone https://github.com/streamlit/streamlit-example.git .

# Install Streamlit requirements
RUN pip3 install -r requirements.txt

# Expose port 8501
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Entry point for running the Streamlit app
ENTRYPOINT ["streamlit", "run", "code_frontend_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
