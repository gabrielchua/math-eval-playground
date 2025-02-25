# Use Python 3.12 slim base image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in same layer to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -r -u 999 appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer (uv installer)
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies using uv's pip
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY questions/ questions/.
COPY .env .

# Copy binaries from /root/.local/bin to /usr/local/bin so all users can run them
RUN cp -r /root/.local/bin/* /usr/local/bin/

# Change ownership of application files to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for Streamlit
EXPOSE 3000

# Command to run the Streamlit app on the specified port
CMD ["streamlit", "run", "app.py", "--server.port=3000"]
