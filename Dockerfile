# ------------------------
# Base Image
# ------------------------
FROM python:3.10

# ------------------------
# Working Directory
# ------------------------
WORKDIR /app

# ------------------------
# Copy requirements
# ------------------------
COPY requirements.txt .

# ------------------------
# Install dependencies
# ------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------
# Copy project files
# ------------------------
COPY . .

# ------------------------
# Expose Port
# ------------------------
EXPOSE 8000

# ------------------------
# Run FastAPI App
# ------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]