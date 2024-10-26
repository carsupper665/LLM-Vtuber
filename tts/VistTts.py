import soundfile
import time
import os
import sys
sys.path.append('tts/src')
os.environ["PYTORCH_JIT"] = "0"
import torch

import tts.src.commons as commons
import tts.src.utils as utils

from .src.models import SynthesizerTrn
from .src.text.symbols import symbols
from .src.text import text_to_sequence


class VistTTS:
    def __init__(self, model_path, config_path, speed) -> None:
        self.callback('VistTTS init...')
        self.speed = speed
        self.hps = utils.get_hparams_from_file(config_path)
        self.model = self._load_model(model_path)

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
            
        soundfile.write(f'./web/audio/{file}', audio, self.hps.data.sampling_rate)

        self.callback(f'VistTTS Synth Done, time used: {(time.time() - stime):.2f}')
        return audio

    def callback(self, msg):
        print(f'[DeBug] [Vist TTS] | {msg}')

if __name__ == "__main__":
    tts = VistTTS(
        model_path=r".\tts\models\paimon6k_390k.pth",
        config_path=r".\tts\models\paimon6k.json",
        speed=1
    )
    tts.generate_audio("你好，歡迎使用 VistTTS 語音服務！")
