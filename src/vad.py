# src/vad.py
import os
import wave
import contextlib
import webrtcvad
import collections
import sys
import numpy as np
import soundfile as sf

def read_wave(path):
    data, sr = sf.read(path, dtype='int16')
    # ensure mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1).astype('int16')
    return data.tobytes(), sr

def write_wave(path, audio, sr):
    sf.write(path, np.frombuffer(audio, dtype='int16'), sr, subtype='PCM_16')

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample (16bit)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / (2 * sample_rate))
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
                triggered = False
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def split_audio_vad(input_path, output_dir, aggressiveness=2):
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = read_wave(input_path)
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(30, audio, sr)
    frames = list(frames)
    segments = vad_collector(sr, 30, 300, vad, frames)
    saved = []
    i = 0
    for seg in segments:
        out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_seg{i}.wav")
        write_wave(out_path, seg, sr)
        saved.append(out_path)
        i += 1
    return saved

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    print("Splitting", args.input, "->", args.output_dir)
    s = split_audio_vad(args.input, args.output_dir)
    print("Saved segments:", s)
