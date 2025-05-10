from flask import Flask, request, jsonify
from flask_cors import CORS
import time, hmac, hashlib, base64, wave, pyaudio, requests, json

# ACRCloud credentials
HOST = 'https://identify-ap-southeast-1.acrcloud.com/v1/identify'
ACCESS_KEY = 'cd14d2757fb14a000cd6016a9ad854dc'
ACCESS_SECRET = '0RuaQqJB6WgmZjzvMSra8CpqIoeKQ6KduQgEQ3Rq'

app = Flask(__name__)
CORS(app)

def record_audio(file_name="recorded.wav", duration=10):
    RATE = 44100
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return file_name

def recognize(audio_path):
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))

    with open(audio_path, 'rb') as f:
        sample_bytes = f.read()

    string_to_sign = '\n'.join([http_method, http_uri, ACCESS_KEY, data_type, signature_version, timestamp])
    sign = base64.b64encode(hmac.new(ACCESS_SECRET.encode(), string_to_sign.encode(), hashlib.sha1).digest()).decode()

    files = {'sample': sample_bytes}
    data = {
        'access_key': ACCESS_KEY,
        'sample_bytes': len(sample_bytes),
        'timestamp': timestamp,
        'signature': sign,
        'data_type': data_type,
        'signature_version': signature_version,
    }

    response = requests.post(HOST, files=files, data=data)
    return response.json()

@app.route('/identify', methods=['GET'])
def identify_song():
    file = record_audio(duration=10)
    result = recognize(file)
    try:
        music_info = result['metadata']['music'][0]
        return jsonify({
            'title': music_info.get('title', 'Unknown'),
            'artists': [a['name'] for a in music_info.get('artists', [])],
            'album': music_info.get('album', {}).get('name', 'Unknown'),
            'release_date': music_info.get('release_date', 'Unknown')
        })
    except:
        return jsonify({'error': 'Could not recognize song'}), 400

if __name__ == "__main__":
    # app.run(port=5000)
    app.run(host='0.0.0.0', port=5000)

