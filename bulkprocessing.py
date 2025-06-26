import whisperx
import torch
from dotenv import load_dotenv
import os
import time
import threading
from datetime import datetime

# --- LOAD .ENV VARIABLES ---
load_dotenv()
your_token = os.getenv("HUGGINGFACE_TOKEN")

if your_token is None:
    raise ValueError("Hugging Face token not found! Please set it in your .env file.")

# --- SETTINGS ---
batch_size = 16
compute_type = "float32"

# --- DEVICE SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- KEEP-ALIVE LOGGER ---
keep_running = True
def keep_alive_logger():
    while keep_running:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"I am still running, don't worry... (10-minute check-in at {current_time})")
        time.sleep(600)

threading.Thread(target=keep_alive_logger, daemon=True).start()

# --- AUDIO PROCESSING FUNCTION ---
def process_audio(audio_file, output_txt_spanish, output_txt_english):
    print(f"\n----- Processing {audio_file} -----")
    audio = whisperx.load_audio(audio_file)

    print("Loading WhisperX model...")
    model = whisperx.load_model("small", device, compute_type=compute_type)

    print("Transcribing Spanish audio...")
    result_spanish = model.transcribe(audio, batch_size=batch_size, language="es")

    print("Translating to English...")
    result_english = model.transcribe(audio, batch_size=batch_size, language="es", task="translate")

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

    print("Aligning Spanish transcription...")
    result_spanish = whisperx.align(result_spanish["segments"], model_a, metadata, audio, device)

    print("Aligning English translation...")
    result_english = whisperx.align(result_english["segments"], model_a, metadata, audio, device)

    print("Loading diarization model...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=your_token, device=device)

    print("Performing diarization...")
    diarize_segments = diarize_model(audio)

    print("Assigning speakers...")
    result_spanish = whisperx.assign_word_speakers(diarize_segments, result_spanish)
    result_english = whisperx.assign_word_speakers(diarize_segments, result_english)

    print(f"Saving Spanish transcription to {output_txt_spanish}...")
    with open(output_txt_spanish, "w", encoding="utf-8") as f:
        for segment in result_spanish["segments"]:
            f.write(f"[{segment.get('speaker', 'Unknown')}]: {segment['text'].strip()}\n")

    print(f"Saving English transcription to {output_txt_english}...")
    with open(output_txt_english, "w", encoding="utf-8") as f:
        for segment in result_english["segments"]:
            f.write(f"[{segment.get('speaker', 'Unknown')}]: {segment['text'].strip()}\n")

    print(f"Done with {audio_file}")

# --- MAIN EXECUTION LOOP ---
audio_jobs = [
    ("audio1.mp3", "transcription1_spanish.txt", "translation1_english.txt"),
    ("audio2.mp3", "transcription2_spanish.txt", "translation2_english.txt"),
    ("audio3.mp3", "transcription3_spanish.txt", "translation3_english.txt")
]

start_time = time.time()
for job in audio_jobs:
    process_audio(*job)
keep_running = False
end_time = time.time()

# --- FINAL STATS ---
total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
print(f"\nAll files processed! 🎉 Total time: {hours}h {minutes}m {seconds}s")
