import argparse
import base64
import json
import os
import time

import cv2
import numpy as np

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = lambda f: os.environ

from openai import OpenAI, AzureOpenAI
import openai
import re

# compare two videos using GPT-4 Vision
def load_credentials(env_file: str) -> dict:
    """
    Load credentials from a .env file or environment.
    """
    creds = dotenv_values(env_file)
    required = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    for key in required:
        creds.setdefault(key, os.getenv(key, ""))
    return creds


def init_vlm_client(env_file: str):
    """
    Initialize the GPT-4 Vision client for Azure or OpenAI.
    """
    creds = load_credentials(env_file)
    if creds.get("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            api_key=creds["AZURE_OPENAI_API_KEY"],
            azure_endpoint=creds["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-02-01"
        )
        return client, {"model": creds["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    client = OpenAI(api_key=creds["OPENAI_API_KEY"])
    return client, {"model": "gpt-4o"}



def build_prompt_content(
    original_bt: str,
    fail_message: str = "",
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    content = []
    header = "You are tasked to debug a behavior tree.  A robot follow a behavior tree, and it failed. The original BT is as follows:\n\n"
    header += original_bt + "\n\n"
    header += "An error detail is as follows:\n\n"
    header += fail_message + "\n\n"
    header += "Please write a behavior tree under the assumption that the robot and environment states remain unchanged at the point of BT failure, so that the BT can be restarted from the failed state. Output only the content within the Tree field, and use only the Nodes that are used in the Original."
    header += "Do not include any additional text or explanations. Do not start with ```Python, ```json, or similar. Return only the BT content."
    header += "Make sure to start with \"PREPARE\" node, which is needed as default."
    content.append({"type": "text", "text": header})

    return content


def query_vlm(client, client_params: dict, prompt_content: list[dict]) -> dict:
    """
    Call the Vision LLM with provided content and return parsed JSON.
    """
    messages = [{"role": "user", "content": prompt_content}]
    params = {**client_params, "messages": messages, "max_tokens": 1000, "temperature": 0.1, "top_p": 0.5}

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**params)
            text = resp.choices[0].message.content
            return text
        except (openai.RateLimitError, openai.APIStatusError) as e:
            print(f"API error: {e} (retrying)")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e} (retrying)")
            time.sleep(1)

    print(f"Failed after {attempt+1} attempts.")
    return {}


def ask_gpt(
    client, client_params: dict,
    original_bt: str,
    fail_message: str,
) -> float:
    """
    Sample frames from both videos, query GPT, and return True if 'first' wins.
    """

    prompt_content = build_prompt_content(original_bt, fail_message)
    result = query_vlm(client, client_params, prompt_content)
    # print(result)
    return result