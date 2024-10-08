import sys
from piper import PiperVoice

def generate_welcome_audio():
    voice = PiperVoice.load("/app/piper/en_GB-alan-medium.onnx")
    text = "Welcome to the WebXR Graph Visualization. Your virtual environment is now ready."
    audio = voice.synthesize(text)
    sys.stdout.buffer.write(audio)

if __name__ == "__main__":
    generate_welcome_audio()