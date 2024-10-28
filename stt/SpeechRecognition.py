import speech_recognition as sr

class SpeechRecognition:
    def __init__(self, language: str, ) -> None:
        self.language = language
        self.r = sr.Recognizer()

    def recognize(self,):
        with sr.Microphone() as source:
            self.callback("speak plz...")

            self.r.adjust_for_ambient_noise(source)

            audio = self.r.listen(source)
        
        try:
            text = self.r.recognize_google(audio, language='zh-CN')
            self.callback(text)
            return text
        except sr.UnknownValueError as e:
            self.callback(f"Unknown Value Error。{e}")
        except sr.RequestError as e:
            self.callback("Request Error: {0}".format(e))

    
    def callback(self, msg):
        print(f'[DeBug] [SpeechRecognition] | {msg}')

if __name__ == "__main__":
    stt = SpeechRecognition('zh-CN')
    stt.recognize()