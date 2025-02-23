#!/usr/bin/env python3
"""
AI Module
==========
Handles AI text generation using Falcon-7B or Llama3.
"""

import logging
import time
import torch
import transformers
import ollama
from transformers import AutoTokenizer


class AIModule:
    def __init__(self, use_falcon=True):
        self.use_falcon = use_falcon
        if use_falcon:
            self.model_id = "tiiuae/falcon-7b-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.pipeline = transformers.pipeline("text-generation", model=self.model_id, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            self.model_id = "llama3"

    def generate_answer(self, prompt):
        start_time = time.time()
        if self.use_falcon:
            response = self.pipeline(prompt, max_length=200, do_sample=True, top_k=10, num_return_sequences=1)[0]['generated_text']
        else:
            response = ollama.chat(model=self.model_id, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip()
        return response
