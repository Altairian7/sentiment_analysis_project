import sounddevice as sd
import scipy.io.wavfile as wav
from TTS.api import TTS
import os

def record_audio(filename="my_voice.wav", duration=5, fs=16000):
    """
    Record your voice for a few seconds and save as a .wav file.
    """
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"✅ Recording saved as {filename}")

def load_tts_model(use_gpu=True):
    """
    Load XTTS-v2 voice cloning model from Hugging Face.
    """
    print("🔁 Loading TTS model (XTTS-v2)...")
    tts = TTS(model_name="coqui/xtts-v2", progress_bar=True, gpu=use_gpu)
    print("✅ TTS model loaded.")
    return tts

def synthesize_speech(tts_model, text, reference_audio="my_voice.wav", output_file="output.wav"):
    """
    Synthesize speech using a reference speaker's voice.
    """
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"{reference_audio} not found.")
    
    print("🗣️ Synthesizing text in your voice...")
    tts_model.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language="en",
        file_path=output_file
    )
    print(f"✅ Synthesized voice saved to: {output_file}")

def main():
    record_audio()  # Record your voice
    tts_model = load_tts_model(use_gpu=True)  # Load TTS model with GPU enabled
    text = input("📝 Enter the text you want to speak: ")
    synthesize_speech(tts_model, text)

if __name__ == "__main__":
    main()
