import os
from pydub import AudioSegment
import speech_recognition as sr

# === Step 1: Convert .ogg to .wav ===
def convert_ogg_to_wav(ogg_path, wav_path):
    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(wav_path, format="wav")
    return wav_path

# === Step 2: Transcribe the audio file ===
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    output_file = "ogg_transcription.txt"

    # Convert to wav if .ogg
    if file_path.lower().endswith(".ogg"):
        wav_path = file_path.replace(".ogg", ".wav")
        print(f"🎧 Converting {file_path} to {wav_path}...")
        file_path = convert_ogg_to_wav(file_path, wav_path)

    # Load and recognize
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-IN")
        print("📝 Transcribed Text:\n", text)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        print(f"✅ Transcription saved to {output_file}")
    except sr.UnknownValueError:
        print("🤷‍♂️ Could not understand the audio")
    except sr.RequestError as e:
        print(f"❌ API error: {e}")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")

# === Run it ===
if __name__ == "__main__":
    audio_path = "./audio/sample.ogg"  # Replace with your .ogg file path
    transcribe_audio(audio_path)
