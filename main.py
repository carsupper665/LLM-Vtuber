    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #                   _oo0oo_
    #                   o8888888o
    #                   88" . "88
    #                   (| -_- |)
    #                   0\  =  /0
    #                 ___/`---'\___
    #               .' \\|     |// '.
    #              / \\|||  :  |||// \
    #             / _||||| -:- |||||- \
    #            |   | \\\  - /// |   |
    #            | \_|  ''\---/''  |_/ |
    #            \  .-\__  '-'  ___/-. /
    #          ___'. .'  /--.--\  `. .'___
    #       ."" '<  `.___\_<|>_/___.' >' "".
    #      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
    #      \  \ `_.   \_ __\ /__ _/   .-` /  /
    #  =====`-.____`.___ \_____/___.-`___.-'=====
    #                    `=---='

    #    此專案被 南無BUG菩薩保佑，不當機，不報錯
    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.staticfiles import StaticFiles
from llm.Interface import llm_interface
from llm.LlamaAPI import llm_api
from tts.tts import tts
from stt.stt import stt
from scripts.PorjectBlesser import blessing
import uvicorn
import argparse
import yaml
import time

import asyncio

class web_app():
    def __init__(self, path=r'.\config.yaml'):
        self.load_cfg(path)
        if self.args.bless:
            blessing()
        system = open('./prompts/' + self.args.prompts, 'r', encoding='utf-8').read()
        self.app = FastAPI()
        self.router = APIRouter()
        self.file = "tts_audio1.mp3"

        # connect to index.html
        self.set_routes()

        self.app.mount("/", StaticFiles(directory="./web", html=True), name="web")
        self.app.mount("/", StaticFiles(directory="./audio"), name="audio")

        self.app.include_router(self.router)

        self.llm = self.init_llm(self.args.llm_url, self.args.api_key, self.args.model, self.args.organization_id, self.args.project_id, system)
        tts_config = self.args.tts
        stt_config = self.args.stt
        
        self.stt = self.init_stt(self.args.stt_mode, stt_config)
        self.tts = self.init_tts(self.args.tts_mode, tts_config.get(self.args.tts_mode, {}))

        self.callback('啟動了耶~')

    def init_llm(self, llm_url, api_key, model, org_id, pro_id, system) -> llm_interface:
        self.callback(f"connect to llm llm_url:{llm_url}")
        # self.callback(f"api_key:{api_key}, model:{model}")
        return llm_api(
        llm_url=llm_url,
        model=model,
        api_key=api_key,
        org=org_id,
        project=pro_id,
        system=system )
    
    def init_tts(self, tts_type, config):
        self.callback(f'TTS type: {tts_type}. Config: {config}')
        return tts.init(tts_type=tts_type, **config)
    
    def init_stt(self, stt_type, config):
        self.callback(f'STT type: {stt_type}. Config: {config}')
        return stt.init(stt_type=stt_type, **config)

    def set_routes(self):
        @self.app.websocket("/llm-ws")    
        async def websocket_handler(websocket: WebSocket,): # 不能使用 self
            await websocket.accept()
            self.callback("WebSocket connection established")
            
            # 持續接收客戶端消息
            while True:
                try:
                    text = ''
                    data = await websocket.receive_text()
                    # self.callback(f"Received: {data} send to llm...")
                    t = self.stt.recognize()
                    self.callback(f"User input:{t}")
                    start = time.time()
                    # res = self.llm.chat_iter(str(data))
                    res = self.llm.chat_iter(str(t))

                    text = res
                    
                    # for chunk in res:
                    #     if chunk:
                    #         text += chunk

                    self.callback(f"Response: {text}")

                    self.tts.generate_audio(str(text), self.file)

                    await websocket.send_text(f"Audio file: ./audio/{self.file}")
                    await websocket.send_text(f"Message res: {text}")

                    if self.file == "tts_audio1.mp3":
                        self.file = "tts_audio2.mp3"
                    else:
                        self.file = "tts_audio1.mp3"

                    self.callback(f'LLM model: {self.args.model}')
                    self.callback(f'TTS mode: {self.args.tts_mode}')
                    self.callback(f'Total latency: {(time.time()-start):.4f}s')
                except Exception as e:
                    self.callback(f"WebSocket error: {e}")
                    self.callback(f"Connection close.")
                    break
            await websocket.close()

    def load_cfg(self, path):
        config = yaml.load(open(path, 'r', encoding='UTF-8'), Loader=yaml.FullLoader)
        self.args = argparse.Namespace(**config)
        self.callback("config loaded")

    def callback(self, msg):
        print(f'[DeBug] [WebSocket] | {msg}')

    def start_server(self):
        uvicorn.run(self.app, host="127.0.0.1", port=self.args.port)

if __name__ == "__main__":
    run = web_app()
    run.start_server()