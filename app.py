import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/api")
async def get_data():
    return {"message": "Hello from port 5006"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5006)