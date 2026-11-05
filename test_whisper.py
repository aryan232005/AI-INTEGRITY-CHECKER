import socket

_original = socket.getaddrinfo

def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only

from faster_whisper import WhisperModel

print("Downloading/loading Faster Whisper tiny model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("FASTER WHISPER MODEL LOADED SUCCESSFULLY")
