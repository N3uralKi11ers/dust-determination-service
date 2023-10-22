import shutil
import os

from fastapi import APIRouter, UploadFile, File

from ...services import dust_determination

router = APIRouter()

@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    
    # folder for data
    data_folder_name = 'data'
    data_folder_path = f'{data_folder_name}/'
    
    # Clean data folder
    create_folder_if_not_exists(data_folder_name)
    delete_files_in_folder(data_folder_path)
    
    with open(data_folder_path + f"{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    dust_determination(data_folder_path, file.filename)
        
    return {"filename": file.filename}


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')