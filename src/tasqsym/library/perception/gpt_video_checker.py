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


def open_image(image_path: str) -> np.ndarray:
    """
    Open an image file and return it as a BGR numpy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return frame

def resize_and_encode(frame: np.ndarray, max_short: int = 768, max_long: int = 2000) -> str:
    """
    Resize a BGR image to fit within max dimensions, encode to JPEG+base64 URI.
    """
    h, w = frame.shape[:2]
    ar = w / h
    if ar >= 1:
        new_w = min(w, max_long)
        new_h = min(int(new_w / ar), max_short)
    else:
        new_h = min(h, max_long)
        new_w = min(int(new_h * ar), max_short)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode('.jpg', resized)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def build_prompt_content(
    prompt: str,
    frames_candidate: list[np.ndarray],
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    content = []
    header = prompt.strip()
    content.append({"type": "text", "text": header})

    for frame in frames_candidate:
        uri = resize_and_encode(frame)
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})

    footer = (
        ""
    )
    content.append({"type": "text", "text": footer})
    return content


def query_vlm(client, client_params: dict, prompt_content: list[dict]) -> dict:
    """
    Call the Vision LLM with provided content and return parsed JSON.
    """
    messages = [{"role": "user", "content": prompt_content}]
    params = {**client_params, "messages": messages, "max_tokens": 50, "temperature": 0.1, "top_p": 0.5}

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
    prompt: str,
    frames_candidate: list[np.ndarray],
) -> float:
    """
    Sample frames from both videos, query GPT, and return True if 'first' wins.
    """

    prompt_content = build_prompt_content(prompt, frames_candidate)
    result = query_vlm(client, client_params, prompt_content)
    # print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Status check using GPT-4 Vision.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--img", default="sample.jpg", help="Image path.")
    parser.add_argument("--prompt", default="Check the robot's progress in grasping an object.", help="Prompt for GPT.")
    args = parser.parse_args()
    client, client_params = init_vlm_client(args.creds)

    frames_candidate = open_image(args.img)
    progress = ask_gpt(
        client, client_params,
        args.prompt,
        [frames_candidate]  # Assuming a single image for simplicity
    )
    print(f"Video progress stage: {progress}")

if __name__ == "__main__":
    main()
