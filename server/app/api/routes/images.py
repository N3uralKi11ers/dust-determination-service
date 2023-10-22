import os

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


@router.get("/before/{image_id}")
async def get_image_before(image_id: int):
    img_data = _get_image(
		folder="X",
		image_id=image_id
	)
    return Response(content=img_data, media_type="image/png")
  

@router.get("/after/{image_id}")
async def get_image_after(image_id: int):
    img_data = _get_image(
		folder="Y",
		image_id=image_id
	)
    return Response(content=img_data, media_type="image/png")
  

def _get_image(folder: str, image_id: int):
    file_path = os.path.join('data', 'res', folder, f'{image_id}.jpg')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    else:
        return None