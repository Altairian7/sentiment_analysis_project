import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the microphone for real-time input
with sr.Microphone() as source:
    print("🎤 Say something! (Press Ctrl+C to stop)")
    
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source)

    try:
        while True:
            # Listen for audio
            audio = recognizer.listen(source)

            try:
                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio, language="en-IN")
                print("📝 You said:", text)
            except sr.UnknownValueError:
                print("🤷‍♂️ Could not understand audio")
            except sr.RequestError as e:
                print(f"❌ Could not request results; {e}")
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
