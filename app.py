from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import torch
import whisper
from langdetect import detect
from pydub import AudioSegment
import cv2
from nudenet import NudeDetector
import re


class VideoTranscriptionLanguageDetector:
    def __init__(self, model_size='base'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model ({model_size}) on {self.device}")
        self.model = whisper.load_model(model_size).to(self.device)

    def extract_audio(self, video_path):
        os.makedirs('temp', exist_ok=True)
        audio = AudioSegment.from_file(video_path)
        audio_path = os.path.join('temp', 'extracted_audio.wav')
        audio.export(audio_path, format='wav')
        return audio_path

    def transcribe_audio(self, audio_path):
        print("Transcribing audio...")
        result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
        return result

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            print(f"Language detection error: {e}")
            return None

    def process_video(self, video_path):
        audio_path = self.extract_audio(video_path)
        transcription_result = self.transcribe_audio(audio_path)
        language = self.detect_language(transcription_result['text'])
        os.remove(audio_path)
        return {
            'transcription': transcription_result['text'],
            'language': language,
            'detected_segments': transcription_result.get('segments', [])
        }


def extract_frames(video_path, output_folder, interval=1.5):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1

    video.release()
    print(f"Saved {saved_frame_count} frames to {output_folder}.")


def classify_images_in_folder(image_folder):
    detector = NudeDetector()
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return False

    image_paths = [os.path.join(image_folder, img) for img in image_files]
    results = detector.detect_batch(image_paths)
    nsfw_labels = {
        "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", 
        "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
    }
    for detections in results:
        if any(d['class'] in nsfw_labels and d['score'] > 0.4 for d in detections):
            return True
    return False


def check_bad_words(transcripted_text):
    bad_words_list = {"damn", "hell", "shit", "fuck", "cunt", "bastard", "dick", "slut"}
    cleaned_text = re.sub(r'[^\w\s]', '', transcripted_text.lower())
    return any(word in bad_words_list for word in cleaned_text.split())


def detect_faces_in_images(image_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Haar Cascade not found.")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return True
    return False


app = FastAPI()
transcription_system = VideoTranscriptionLanguageDetector()


@app.post("/analyze-video/")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_video"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_folder = "temp_frames"
        os.makedirs(output_folder, exist_ok=True)
        extract_frames(video_path, output_folder, interval=1.5)

        nudity_found = classify_images_in_folder(output_folder)
        faces_detected = detect_faces_in_images(output_folder)
        transcription_result = transcription_system.process_video(video_path)
        bad_words_found = check_bad_words(transcription_result["transcription"])

        for frame_file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, frame_file))
        os.rmdir(output_folder)
        os.remove(video_path)
        os.rmdir(temp_dir)

        return JSONResponse(content={
            "video_path": file.filename,
            "transcription": transcription_result["transcription"],
            "language": transcription_result["language"],
            "human_face_detected": faces_detected,
            "nsfw_detected": nudity_found,
            "inappropriate_language_detected": bad_words_found
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)