class stt():
    @staticmethod
    def init(stt_type, **args) -> None:
        if stt_type == "sr":
            from .SpeechRecognition import SpeechRecognition
            return SpeechRecognition(args.get('language'))
        else:
            raise ValueError(f"[DeBug] [STT] | Unknown STT engine type: {stt_type}")
        

