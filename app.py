# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from main import DataQualityEngine  # your engine file
import io

app = FastAPI(title="Data Quality Engine API", version="1.0")


@app.post("/api")
async def analyze_file(file: UploadFile = File(...), sample_size: int = 5):
    """
    Upload a CSV or Excel file and get Data Quality report.
    """
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")

        engine = DataQualityEngine(df, name=file.filename, sample_size=sample_size)
        report = engine.run_all()

        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# To run the server:
# uvicorn app:app --reload --port 8000
