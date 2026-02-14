import hashlib
import json
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from glm5_client import GLM5Error, call_glm5


class BadRequest(Exception):
    pass


def parse_content_type(value: str) -> tuple[str, dict[str, str]]:
    main, *parts = [p.strip() for p in value.split(';') if p.strip()]
    params: dict[str, str] = {}
    for part in parts:
        if '=' in part:
            k, v = part.split('=', 1)
            params[k.strip().lower()] = v.strip().strip('"')
    return main.lower(), params


def parse_multipart(body: bytes, boundary: str) -> tuple[dict[str, str], bytes]:
    delimiter = f"--{boundary}".encode("utf-8")
    for part in body.split(delimiter):
        part = part.strip()
        if not part or part == b"--":
            continue

        if b"\r\n\r\n" not in part:
            continue

        raw_headers, content = part.split(b"\r\n\r\n", 1)
        content = content.rstrip(b"\r\n")
        headers: dict[str, str] = {}
        for line in raw_headers.decode("utf-8", errors="ignore").split("\r\n"):
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()

        disposition = headers.get("content-disposition", "")
        _, disp_params = parse_content_type(disposition.replace("form-data", "form-data;", 1))
        if disp_params.get("name") == "file":
            filename = disp_params.get("filename") or "upload.bin"
            return {
                "filename": filename,
                "content_type": headers.get("content-type", "application/octet-stream"),
            }, content

    raise BadRequest("Missing 'file' field")


def sniff_image_metadata(payload: bytes) -> dict[str, object]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        width, height = struct.unpack(">II", payload[16:24])
        image_type = "png"
    elif payload.startswith(b"\xff\xd8"):
        image_type = "jpeg"
        width = height = None
        i = 2
        while i + 9 < len(payload):
            if payload[i] != 0xFF:
                i += 1
                continue
            marker = payload[i + 1]
            if marker in {0xC0, 0xC2}:
                block_len = struct.unpack(">H", payload[i + 2:i + 4])[0]
                if i + 2 + block_len <= len(payload):
                    height, width = struct.unpack(">HH", payload[i + 5:i + 9])
                    break
            if marker in {0xD8, 0xD9}:
                i += 2
            else:
                block_len = struct.unpack(">H", payload[i + 2:i + 4])[0]
                i += 2 + block_len
        if width is None or height is None:
            raise BadRequest("Unsupported or malformed JPEG image")
    else:
        raise BadRequest("Unsupported image type. Use PNG or JPEG")

    return {
        "image_type": image_type,
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(width / height, 4) if height else None,
    }


class DiagramHandler(BaseHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        self._write_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/diagram/analyze":
            self.handle_analyze()
            return
        if parsed.path == "/api/v1/diagram/interpret":
            self.handle_interpret(parsed.query)
            return
        self._write_json(404, {"error": "Not found"})

    def handle_analyze(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        main_type, params = parse_content_type(content_type)
        if main_type != "multipart/form-data" or "boundary" not in params:
            self._write_json(400, {"error": "Expected multipart/form-data with file field"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise BadRequest("Empty request body")
            body = self.rfile.read(length)
            file_meta, payload = parse_multipart(body, params["boundary"])
            if not payload:
                raise BadRequest("Uploaded file is empty")

            image_meta = sniff_image_metadata(payload)
            sha256 = hashlib.sha256(payload).hexdigest()

            self._write_json(
                200,
                {
                    "filename": file_meta["filename"],
                    "content_type": file_meta["content_type"],
                    "size_bytes": len(payload),
                    "sha256": sha256,
                    "endpoint": "/api/v1/diagram/analyze",
                    **image_meta,
                },
            )
        except BadRequest as exc:
            self._write_json(400, {"error": str(exc)})

    def handle_interpret(self, query: str) -> None:
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._write_json(400, {"error": "Expected application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise BadRequest("Empty request body")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))

            prompt = payload.get("prompt", "")
            image_b64 = payload.get("image_b64")
            api_key = payload.get("api_key")

            result = call_glm5(prompt=prompt, api_key=api_key, image_b64=image_b64)
            query_params = parse_qs(query)
            include_raw = query_params.get("raw", ["false"])[0].lower() == "true"

            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            output_text = message.get("content", "")
            reasoning = message.get("reasoning_content")

            response = {
                "endpoint": "/api/v1/diagram/interpret",
                "model": result.get("model"),
                "content": output_text,
                "reasoning_content": reasoning,
            }
            if include_raw:
                response["raw"] = result
            self._write_json(200, response)
        except (BadRequest, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": str(exc)})
        except GLM5Error as exc:
            self._write_json(502, {"error": str(exc)})


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), DiagramHandler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
