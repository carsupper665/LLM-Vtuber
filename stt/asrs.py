import time
import wave
import pyaudio
import numpy as np

from stt.rapid_paraformer import RapidParaformer


class ASRS():
    def __init__(self, config_path):
        self._print('Initializing ASR Service...')
        self.paraformer = RapidParaformer(config_path)

    def infer(self, wav_path):
        stime = time.time()
        result = self.paraformer(wav_path)
        self._print('ASR Result: %s. time used %.2f.' % (result, time.time() - stime))
        return result[0]
    
    def _print(self, msg):
        print('[Debug] [ASRService]', msg)

    def recognize(self):
        total_time = time.time()
        # Recording configuration
        chunk = 1024
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        rate = 16000  # 16kHz sample rate
        threshold = 500  # Silence threshold to stop recording
        silence_duration_limit = 1.5  # Stop after 1.5 seconds of silence
        min_recording_time = 3  # Minimum recording time in seconds
        max_recording_time = 10  # Maximum recording time in seconds

        p = pyaudio.PyAudio()
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        self._print("Recording...")
        frames = []
        silence_duration = 0
        start_time = time.time()

        while True:
            data = stream.read(chunk)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.abs(audio_data).mean()

            if audio_level < threshold:
                silence_duration += chunk / rate
            else:
                silence_duration = 0

            elapsed_time = time.time() - start_time

            if elapsed_time >= max_recording_time:
                self._print("Maximum recording time reached. Stopping recording.")
                break

            if silence_duration >= silence_duration_limit and elapsed_time >= min_recording_time:
                self._print("Silence detected. Stopping recording.")
                break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data as a WAV file
        wav_path = "/recorded_audio.wav"
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Run inference
        result = self.infer(wav_path)
        self._print('Total time used %.2f.' % (time.time() - total_time))
        return result
    
if __name__ == '__main__':
    config_path = 'stt/resources/config.yaml'

    service = ASRS(config_path)

    result = service.recognize()
    print(result)