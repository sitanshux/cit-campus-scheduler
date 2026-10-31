from fastapi import FastAPI, HTTPException

from app.core.dataset import (
    load_dataset,
    get_dataset_summary,
)
from app.scheduling.routes import router as schedule_router

app = FastAPI(
    title="CIT Campus Scheduler API",
    description="Campus Scheduling and Venue Management System",
    version="1.0.0",
)

app.include_router(schedule_router)


@app.get("/")
def root():
    return {
        "message": "CIT Campus Scheduler API is running",
        "status": "healthy",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CIT Campus Scheduler Backend",
    }


@app.get("/api/dataset/summary")
def dataset_summary():
    try:
        return get_dataset_summary()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/dataset")
def dataset_info():
    try:
        dataset = load_dataset()

        return {
            "dataset_id": dataset["dataset_id"],
            "institute": dataset["institute"],
            "calendar": dataset["calendar"],
            "counts": get_dataset_summary()["counts"],
        }

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))
