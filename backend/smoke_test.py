import base64
import json
import threading
import time
import urllib.request
import uuid

import server
from server import run_server


def make_multipart_body(field_name: str, filename: str, content_type: str, data: bytes, boundary: str) -> bytes:
    lines = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"',
        f"Content-Type: {content_type}",
        "",
    ]
    head = "\r\n".join(lines).encode("utf-8") + b"\r\n"
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + data + tail


def main() -> None:
    port = 8090
    thread = threading.Thread(target=run_server, kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    thread.start()
    time.sleep(0.4)

    # Analyze endpoint (1x1 transparent PNG)
    png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Bf8YAAAAASUVORK5CYII=")
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = make_multipart_body("file", "diagram.png", "image/png", png_bytes, boundary)

    analyze_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/diagram/analyze",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urllib.request.urlopen(analyze_req, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assert payload["filename"] == "diagram.png"
    assert payload["size_bytes"] == len(png_bytes)
    assert payload["image_type"] == "png"
    assert payload["width"] == 1
    assert payload["height"] == 1

    # Interpret endpoint (mock GLM5 call to avoid external dependency during smoke test)
    def fake_call_glm5(prompt: str, api_key=None, image_b64=None):
        return {
            "model": "z-ai/glm5",
            "choices": [
                {
                    "message": {
                        "content": f"Mocked interpretation for: {prompt[:20]}",
                        "reasoning_content": "mock-reasoning",
                    }
                }
            ],
        }

    server.call_glm5 = fake_call_glm5  # type: ignore[assignment]

    interpret_body = json.dumps(
        {
            "prompt": "Summarize the uploaded diagram in 3 bullets",
            "image_b64": base64.b64encode(png_bytes).decode("utf-8"),
        }
    ).encode("utf-8")

    interpret_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/diagram/interpret",
        method="POST",
        data=interpret_body,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(interpret_req, timeout=5) as response:
        interpret_payload = json.loads(response.read().decode("utf-8"))

    assert interpret_payload["endpoint"] == "/api/v1/diagram/interpret"
    assert interpret_payload["model"] == "z-ai/glm5"
    assert "Mocked interpretation" in interpret_payload["content"]

    print("Smoke test passed")
    print({"analyze": payload, "interpret": interpret_payload})


if __name__ == "__main__":
    main()
