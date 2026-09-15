import fastapi
from server import Server, PredictionResponse
from typing import List
app=fastapi.FastAPI()
server=Server()


@app.post("/make_predict", response_model=List[PredictionResponse])
async def make_predict(files:list[fastapi.UploadFile]):
    if server.agent is None or server.classifier is None:
        raise fastapi.HTTPException(status_code=503, detail="Model is not available")
    files_bytes=[await file.read() for file in files]
    return server.predict(files_bytes)






