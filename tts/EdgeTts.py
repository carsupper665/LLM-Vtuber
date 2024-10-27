import sys
import os
import edge_tts as tts

from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class EdgeTTS():
    def __init__(self, voice, pitch):
        self.voice = voice
        self.pitch = pitch

    def generate_audio(self, text, file):
        
        file_name = f'./web/audio/{file}'
        self.audio_file = file

        self.remove(file_name)

        try:
            res = tts.Communicate(text, self.voice, pitch=str(self.pitch), )
            res.save_sync(file_name)
            self.callback('tts generated')
        except Exception as e:
            self.callback(
                f"No audio was received. Error: {e}"
            )
            return None

        return file_name
    
    def callback(self, msg):
        print(f'[DeBug] [Edge TTS] | {msg}')

    def remove(self, path):
        if os.path.exists(path):
            os.remove(f'./web/audio/{self.audio_file}')

def test():
    tts = EdgeTTS('en-US-EmmaMultilingualNeural', '+5Hz')
    text = str('''Hey there, VTuber! Just wanted to drop you a line from your friendly neighborhood AI bot. So, what's new in the world of online streaming? Still stuck behind that screen 24/7 or have 
you managed to "get out and touch some grass" lately? Kidding aside, I'm here for you, and we can chat about anything from gaming tips to life hacks. Just remember: a virtual presence doesn't mean you can't get some fresh air!''')
    tts.generate_audio(text)
    tts.callback('Test passed')

if __name__ == '__main__':
    test()