import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000          # sample rate
seconds = 5         # how long to record

print("Speak now...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
sd.wait()
write("input.wav", fs, audio)
print("Saved input.wav")