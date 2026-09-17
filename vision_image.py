"""Read bounded image attachments for the vision endpoint; standard library only."""
import base64
from pathlib import Path


def image_data_url(path: str) -> str:
    limit = 10 * 1024 * 1024
    try:
        with Path(path).open("rb") as source:
            data = source.read(limit + 1)
    except OSError:
        raise ValueError("Cannot read the attached image. Select the file again.") from None
    if len(data) > limit:
        raise ValueError("Image exceeds 10 MB. Choose a smaller image.")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("Choose a PNG, JPEG, or WebP image.")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
