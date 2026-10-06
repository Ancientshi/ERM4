"""Remote chat transport shared by optional prompt-based module adapters."""
import asyncio
import os
import requests


def GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0, historical_qa=None, siliconflow=False, api_key=None):
    """Call a chat completion API without modifying or printing credentials."""
    env_name = "SILICONFLOW_API_KEY" if siliconflow else "OPENAI_API_KEY"
    key = api_key if api_key is not None else os.environ.get(env_name)
    if not key:
        raise ValueError(f"Set {env_name} before making LLM requests.")
    url = ("https://api.siliconflow.cn/v1/chat/completions" if siliconflow
           else "https://api.openai.com/v1/chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    messages = []
    for question, answer in historical_qa or []:
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}])
    messages.append({"role": "user", "content": prompt})
    data = {"model": model_name, "messages": messages, "temperature": t, "n": 1}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
    except requests.RequestException:
        raise RuntimeError("LLM request failed (connection or timeout); no response was recorded.") from None
    if response.status_code != 200:
        # Do not log response bodies: they may contain prompts or provider details.
        raise RuntimeError(f"LLM request failed with HTTP {response.status_code}.")
    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("LLM response did not contain chat completion content.") from None
    if not isinstance(answer, str):
        raise RuntimeError("LLM response content must be text.")
    return answer


async def async_GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0, historical_qa=None):
    # Reuse the same transport and error handling without requiring aiohttp.
    return await asyncio.to_thread(GPT_QA, prompt, model_name, t, historical_qa)
