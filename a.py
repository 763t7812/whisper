from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
# Function to transcribe audio using Whisper
async def transcribe_audio(audio_path):
    model = whisper.load_model("large")  # Adjust model size as needed
    result = model.transcribe(audio_path,fp16=False)
    return result["text"]

@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(await file.read())
            audio_path = tmp_audio.name

        # Transcribe audio to text
        text = await transcribe_audio(audio_path)

        return JSONResponse(status_code=200, content={"transcription": text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})