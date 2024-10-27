# This file is part of LLM-Vtuber
# Reference source https://github.com/t41372/Open-LLM-VTuber?tab=readme-ov-file

# MIT License

# Copyright (c) 2024 Yi-Ting Chiu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Iterator
import json
import time
import os
from openai import OpenAI
from .Interface import llm_interface

class llm_api(llm_interface):
    def __init__(
        self,
        llm_url: str,
        model: str,
        system: str,
        org: str = "None",
        project: str = "None",
        api_key: str = "None",):
        import requests

        self.llm_url = llm_url
        self.llm_url += 'v1/'
        
        self.model = model
        self.system = system
        self.callback = self.constom_callback
        self.memory = []

        try:
            re = requests.get(llm_url)
            if re.status_code == 200:
                self.callback(f'Post status code: {re.status_code}')
                self.callback(f'url:{self.llm_url}')
                
                self.client= OpenAI(base_url=self.llm_url, api_key=api_key)
            else:
                self.callback(f'connect error: {re.status_code}')
                self.callback(f'Ollama is not running')
                self.callback(f'Use OpenAI')
                self.callback(f'api_key:{api_key}')
                self.callback(f'model:{model}')
                self.callback(f'org id:{org}')
                self.callback(f'project id:{project}')
                self.client = OpenAI(
                api_key=api_key,
                organization=org,
                project=project
            )
        except requests.exceptions.RequestException as e:
            self.callback(f"post error: {e}")
            self.callback(f'Ollama is not running')
            self.callback(f'Use OpenAI')
            self.callback(f'api_key:{api_key}')
            self.callback(f'model:{model}')
            self.callback(f'org id:{org}')
            self.callback(f'project id:{project}')
            self.client = OpenAI(
                api_key=api_key,
                organization=org,
                project=project
            )
        
        self.__set_system(self.system)
        self.callback("llm init")

    def __set_system(self, system):
        """
        Set the system prompt
        system: str
            the system prompt
        """
        if not os.path.exists('mem.json'):
            self.system = system
            self.memory.append(
                {
                    "role": "system",
                    "content": system,
                }
            )
            self.callback(f'memory creat')
        else:
            with open('mem.json', 'r') as file:
                self.memory = json.load(file)
            self.system = self.memory
            self.callback(f'memory read')

    def chat_iter(self, prompt: str) -> Iterator[str]:
        start_time = time.time()

        self.memory.append(
            {
            "role": "user",
            "content": prompt,
            }
        )
        response = []
        try:
            self.callback(f'User Input: {prompt}')
            self.callback('LLM generate...')
            response = self.client.chat.completions.create(
                messages=self.memory,
                model=self.model,
                stream=True,
            )
        except Exception as e:
            self.callback(f"Error when response: {str(e)}")
            # yield f"Someone tell Jim there's a problem with my AI."
            return f"Someone tell Jim there's a problem with my AI."
        
        # 生成器，用於迭代生成和存儲回應
        # yield from self._stream_and_store_response(response)
        res = self._stream_and_store_response(response)
        self.callback(f'LLM latency: {(time.time()-start_time):.4f}s')
        return res

    def _stream_and_store_response(self, response) -> Iterator[str]:
        complete_response = ""
         # 迭代獲取每個回應的部分
        for chunk in response:
            # 有時候 delta.content 可能為 None，所以我們確保它至少是一個空字符串
            content = chunk.choices[0].delta.content or ""
            # yield content  # 流式返回每個段落
            complete_response += content  # 累積完整回應

        # 將完整的 AI 回應存儲到記憶中
        self.memory.append(
            {
                "role": "assistant",
                "content": complete_response,
            }
        )
        
        self._save_mem()
        return complete_response

    def _save_mem(self):
        fn = 'mem.json'
        try:
            with open(fn, 'w') as file:
                json.dump(self.memory, file)
            print()
            self.callback(f"Memory saved to {fn}")
        except Exception as e:
            print()
            self.callback(f"Error saving memory to {fn}: {str(e)}")

    def handle_interrupt(self, heard_response: str) -> None:
        """
        當 LLM 被用戶中斷時調用這個函數。
        這個函數需要讓 LLM 知道它被中斷了，並讓它知道用戶只聽到了 heard_response 中的內容。
        這個函數應該：
        - 更新 LLM 的記憶（僅保留 heard_response 而不是生成的完整回應），並讓它知道它在那個點被中斷了。
        - 向 LLM 發出中斷信號。

        參數:
        - heard_response (str): LLM 在被中斷前的最後回應。用戶在中斷前只能聽到的內容。
        """
        if self.memory[-1]["role"] == "assistant":
            self.memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self.memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self.memory.append(
            {
                "role": "system",
                "content": "[Interrupted by user]",
            }
        )
    
    def constom_callback(self, msg):
        print(f"[DeBug] [LlmAPI] | {msg}")
        
def callback(msg):
    print(f"[DeBug] [LlmAPI] | {msg}")

if __name__ == "__main__":
    # Test
    llm = llm_api(
        llm_url="http://localhost:11434/v1",
        model="llama3:latest",
        system='You are a sarcastic AI chatbot who loves to the jokes "Get out and touch some grass" 愛說中文',
        # callback=callback,
        )
    print("\n(Press Ctrl+C to exit.)")
    while True:
        
        chat_complet = llm.chat_iter(input("User inputer>> "))
        print('Ollama response: ', end="")
        for chunk in chat_complet:
            if chunk:
                print(chunk, end="")

    
