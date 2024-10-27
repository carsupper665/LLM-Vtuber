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
import abc
from typing import Iterator


class llm_interface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def chat_iter(self, prompt: str) -> Iterator[str]:
        """
        Sends a chat prompt to an agent and return an iterator to the response.
        This function will have to store the user message and ai response back to the memory.

        Parameters:
        - prompt (str): The message or question to send to the agent.

        Returns:
        - Iterator[str]: An iterator to the response from the agent.
        """
        raise NotImplementedError

    def handle_interrupt(self, heard_response: str) -> None:
        """
        This function will be called when the LLM is interrupted by the user.
        The function needs to let the LLM know that it was interrupted and let it know that the user only heard the content in the heard_response.
        The function should either (consider that some LLM provider may not support editing past memory):
        - Update the LLM's memory (to only keep the heard_response instead of the full response generated) and let it know that it was interrupted at that point.
        - Signal the LLM about the interruption.

        Parameters:
        - heard_response (str): The last response from the LLM before it was interrupted. The only content that the user can hear before the interruption.
        """
        raise NotImplementedError