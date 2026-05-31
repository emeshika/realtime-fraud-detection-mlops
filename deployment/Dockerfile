# Step 1: Use an official lightweight Python runtime as a parent image
FROM python:3.12-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container at /app
COPY requirements.txt .

# Step 4: Install the required packages without using cache (to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all the necessary project directories into the container
COPY ./api /app/api
COPY ./models /app/models
COPY ./config /app/config

# Step 6: Expose the port that FastAPI will run on
EXPOSE 8000

# Step 7: Command to run the FastAPI application using Uvicorn when the container starts
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]