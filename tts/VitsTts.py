import os
import sys
sys.path.append('tts/src')
os.environ["PYTORCH_JIT"] = "0"
import soundfile
import time
import torch
import numpy as np
import onnxruntime

import tts.src.commons as commons
import tts.src.utils as utils

from transformers import BertTokenizer
from .src.models import SynthesizerTrn
from .src.text.symbols import symbols
from .src.text import text_to_sequence

class VitsTTS:
    def __init__(self, model_path, config_path, speed, ifsentiment, sentiment_path='') -> None:
        self.callback('VistTTS init...')
        self.sentiment = None
        if ifsentiment == True:
            self.sentiment = SentimentEngine(f'./tts/se_models/{sentiment_path}')
        self.speed = speed
        self.hps = utils.get_hparams_from_file(f'./tts/models/{config_path}')
        self.model = self._load_model(f'./tts/models/{model_path}')
        

    def _load_model(self, model_path):
        model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        ).cuda()
        model.eval()  # 設定為評估模式
        utils.load_checkpoint(model_path, model, None)  # 加載模型參數
        return model    

    @staticmethod
    def _text_to_tensor(text, hps):
        """將輸入文字轉換為模型所需的 Tensor 格式。"""
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)  # 在符號間插入空白
        return torch.LongTensor(text_norm)
    
    def generate_audio(self, text, file):
        self.callback(f"Processing text: {text}")
        stime = time.time()
        text = text.replace('~', '！')
        stn_tst = self._text_to_tensor(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = self.model.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.2, length_scale=self.speed)[0][
                0, 0].data.cpu().float().numpy()
            
        if self.sentiment != None:
            soundfile.write(f'./web/audio/{file}', audio, self.hps.data.sampling_rate)
            # with open(f'./web/audio/{file}', 'rb') as f:
            #     senddata = f.read()
            se = self.sentiment.infer(text)
            # senddata += b'?!'
            # senddata += b'%i' % se
            # with open(f'./web/audio/{file}', 'wb') as f:
            #     f.write(senddata)
        else:
            soundfile.write(f'./web/audio/{file}', audio, self.hps.data.sampling_rate)

        self.callback(f'VistTTS Synth Done, time used: {(time.time() - stime):.2f}')
        return audio

    def callback(self, msg):
        print(f'[DeBug] [Vist TTS] | {msg}')

class SentimentEngine():
    def __init__(self, model_path):
        self.callback('Initializing Sentiment Engine...')
        onnx_model_path = model_path

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def infer(self, text):
        tokens = self.tokenizer(text, return_tensors="np")
        input_dict = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        # Convert input_ids and attention_mask to int64
        input_dict["input_ids"] = input_dict["input_ids"].astype(np.int64)
        input_dict["attention_mask"] = input_dict["attention_mask"].astype(np.int64)
        logits = self.ort_session.run(["logits"], input_dict)[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted = np.argmax(probabilities, axis=1)[0]
        self.callback(f'Sentiment Engine Infer: {predicted}')
        return predicted
    
    def callback(self, msg):
        print(f'[DeBug] [Sentiment Infer] | {msg}')

if __name__ == "__main__":
    tts = VitsTTS(
        model_path=r".\tts\models\paimon6k_390k.pth",
        config_path=r".\tts\models\paimon6k.json",
        speed=1,
        ifsentiment=True,
        sentiment_path="paimon_sentiment.onnx"
    )
    tts.generate_audio("你好，歡迎使用 VistTTS 語音服務！")
