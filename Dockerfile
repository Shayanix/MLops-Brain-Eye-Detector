FROM python:3.9-slim

WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY config.yaml .
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.predict.app:app", "--host", "0.0.0.0", "--port", "8000"]
