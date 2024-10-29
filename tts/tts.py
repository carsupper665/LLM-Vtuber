# from typing import Type

class tts:
    @staticmethod
    def init(tts_type, **args):
        if tts_type == 'edgeTTS':
            from .EdgeTts import EdgeTTS
            return EdgeTTS(args.get('voice'), args.get('pitch'))
        elif tts_type == 'vitsTTS':
            from .VitsTts import VitsTTS
            return VitsTTS(args.get('model_path'), args.get('config_path'), args.get('speed'), args.get('sentiment'), args.get('sentiment_model_path'))
        else:
            raise ValueError(f"[DeBug] [TTS] | Unknown TTS engine type: {tts_type}")