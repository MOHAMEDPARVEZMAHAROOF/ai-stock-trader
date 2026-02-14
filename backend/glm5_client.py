import json
import os
import urllib.error
import urllib.request
from typing import Any

GLM5_BASE_URL = "https://integrate.api.nvidia.com/v1"
GLM5_MODEL = "z-ai/glm5"


class GLM5Error(RuntimeError):
    pass


def call_glm5(prompt: str, api_key: str | None = None, image_b64: str | None = None) -> dict[str, Any]:
    if not prompt.strip():
        raise GLM5Error("Prompt cannot be empty")

    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise GLM5Error("Missing NVIDIA_API_KEY")

    message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64:
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    payload = {
        "model": GLM5_MODEL,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 2048,
        "stream": False,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        },
    }

    req = urllib.request.Request(
        f"{GLM5_BASE_URL}/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise GLM5Error(f"GLM5 request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise GLM5Error(f"GLM5 network error: {exc}") from exc
