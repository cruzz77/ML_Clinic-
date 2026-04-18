import os
import json
import requests
from groq import Groq
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_api_key(key_name: str) -> str:
    """Helper to get key from st.secrets or environment."""
    try:
        # Check if secrets exist and contains the key
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        # Streamlit may raise an error if secrets.toml is completely missing
        pass
    return os.getenv(key_name, "")

class LLMHandler:
    def __init__(self):
        self.groq_key = get_api_key("GROQ_API_KEY")
        self.hf_token = get_api_key("HF_API_TOKEN")
        
        # Diagnostic print (visible in terminal)
        print(f"DEBUG: Groq Key Loaded: {bool(self.groq_key)}")
        print(f"DEBUG: HF Token Loaded: {bool(self.hf_token)}")
        
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)
        else:
            self.groq_client = None

    def call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API (Primary)."""
        if not self.groq_client:
            raise Exception("Groq API key missing.")
            
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content

    def call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
        """Call Hugging Face Inference API (Fallback)."""
        if not self.hf_token:
            raise Exception("HuggingFace API token missing.")
            
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        response = requests.post(API_URL, headers=headers, json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000, "return_full_text": False}
        })
        
        if response.status_code != 200:
            raise Exception(f"HF API Error: {response.text}")
            
        result = response.json()
        # HF returns a list or direct dict depending on model
        text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
        
        # Attempt to extract JSON if model returned prose
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        return text

    def generate_report(self, system_prompt: str, user_prompt: str) -> str:
        """Try Groq first, then fallback to HuggingFace."""
        try:
            return self.call_groq(system_prompt, user_prompt)
        except Exception as e:
            st.warning(f"Groq API failed, falling back to HuggingFace: {str(e)}")
            try:
                return self.call_huggingface(system_prompt, user_prompt)
            except Exception as e2:
                raise Exception(f"Both LLM providers failed. Final error: {str(e2)}")
