# Video Content Analysis

This project is a FastAPI-based web service that processes video files to extract information and perform various analyses, including transcription, language detection, nudity detection, face detection, and inappropriate language filtering. This project leverages the Whisper model for transcription and several other libraries to analyze video content for safety and appropriateness.

## Features

- **Audio Extraction and Transcription**: Extracts audio from video files and generates transcriptions using Whisper.
- **Language Detection**: Detects the language of the transcribed text using the `langdetect` library.
- **Frame Extraction**: Extracts frames from videos at specified intervals to analyze image content.
- **Nudity Detection**: Scans extracted frames for explicit or NSFW content using the NudeDetector library.
- **Face Detection**: Detects faces in video frames using OpenCV's pre-trained Haar cascades.
- **Inappropriate Language Filtering**: Checks transcriptions for inappropriate language from a predefined list of offensive words.

## Dependencies
- FastAPI: Web framework used to create the API.
- Uvicorn: ASGI server to run the FastAPI application.
- Whisper: OpenAI's Whisper model for speech-to-text transcription.
- langdetect: Language detection library used to detect the language of transcriptions.
- pydub: For audio processing (extracting audio from video).
- opencv-python: For image processing and face detection in frames.
- nudenet: For nudity detection in images.
- torch: Deep learning library used for loading and running the Whisper model.
