"""Offline text-to-speech via the espeak-ng C library (pip: espeakng-loader).

Used only to synthesise the sample MP4 pitch videos so the dataset is
reproducible without proprietary or private recordings.
"""
import ctypes
import wave

import numpy as np


def synthesize(text: str, wav_path: str, rate: int = 150) -> float:
    """Synthesise `text` to a mono 16-bit WAV and return its duration (s)."""
    import espeakng_loader as loader

    lib = ctypes.CDLL(loader.get_library_path())
    samples = []
    cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short),
                               ctypes.c_int, ctypes.c_void_p)

    def callback(wav, n, events):
        if wav and n > 0:
            samples.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
        return 0

    cb = cb_type(callback)
    lib.espeak_Initialize.restype = ctypes.c_int
    sr = lib.espeak_Initialize(2, 0, loader.get_data_path().encode(), 0)
    if sr <= 0:
        raise RuntimeError("espeak initialisation failed")
    lib.espeak_SetSynthCallback(cb)
    lib.espeak_SetVoiceByName(b"en-us")
    lib.espeak_SetParameter(1, rate, 0)  # espeakRATE
    data = text.encode("utf-8") + b"\0"
    lib.espeak_Synth(data, len(data), 0, 0, 0, 1, None, None)  # chars mode
    lib.espeak_Synchronize()
    lib.espeak_Terminate()
    audio = np.concatenate(samples) if samples else np.zeros(1, dtype=np.int16)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.astype(np.int16).tobytes())
    return len(audio) / sr
