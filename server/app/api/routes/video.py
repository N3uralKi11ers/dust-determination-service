from fastapi import APIRouter, UploadFile

router = APIRouter()

@router.post("/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}