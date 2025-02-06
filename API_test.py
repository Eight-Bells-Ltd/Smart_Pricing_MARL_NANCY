import uvicorn
from API import smart_pricing_api  # Import the FastAPI app from your main script

if __name__ == "__main__":
    uvicorn.run(smart_pricing_api, host="127.0.0.1", port=8000)
