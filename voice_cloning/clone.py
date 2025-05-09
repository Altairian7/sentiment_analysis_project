pip install TTS
pip install resemblyzer
pip install pydub sounddevice scipy


import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename="voice_sample.wav", duration=10, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Saved to {filename}")

record_voice()



from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

def get_voice_embedding(wav_path="voice_sample.wav"):
    encoder = VoiceEncoder()
    wav = preprocess_wav(Path(wav_path))
    embed = encoder.embed_utterance(wav)
    return embed
