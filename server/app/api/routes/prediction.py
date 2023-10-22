from fastapi import APIRouter
import pandas as pd

from ...models import FrameBase, Frames

router = APIRouter()


@router.get("/")
def get_frames() -> Frames:
    frames = Frames(frames=[])
    df = pd.read_csv('data/data_time.csv')
    for _, row in df.iterrows():
        frame = FrameBase(
			name=str(row['name']),
   			time=float(row['time']),
			percent=float(row['proc'])
		)
        frames.frames.append(frame)
    return frames


@router.get("/count")
def get_frames_count() -> int:
	df = pd.read_csv('data/data_time.csv')
	num_of_rows = df.shape[0]
	return num_of_rows