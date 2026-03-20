import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    # Read the image file in binary mode and encode it as a base64 string.
    # base64.b64encode() returns bytes, so we decode to a UTF-8 string afterwards.
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # DialModelClient wraps the DIAL /chat/completions HTTP endpoint.
    # The {model} placeholder in DIAL_CHAT_COMPLETIONS_ENDPOINT is replaced
    # with the deployment_name, producing the final URL:
    #   https://ai-proxy.lab.epam.com/openai/deployments/gpt-4o/chat/completions
    # GPT-4o is a vision-capable model; swap it for "claude-3-sonnet", etc. to
    # try other vendors - DIAL Core adapts the request format for each one.
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # ── Approach A: base64 data URL ──────────────────────────────────────────
    # The OpenAI vision spec sends images as a data URL embedded in the JSON body:
    #   "data:<mime_type>;base64,<base64_encoded_bytes>"
    # The entire image travels inside the request - no separate file upload needed.
    # Downside: large images significantly increase the request payload size.
    base64_data_url = f"data:image/png;base64,{base64_image}"

    # ContentedMessage extends Message so that `content` is a *list* of parts
    # rather than a plain string.  This follows the OpenAI multimodal message spec.
    # Supported part types:
    #   TxtContent(text=...) - a text snippet
    #   ImgContent(image_url=ImgUrl(url=...)) - an image reference (data URL or HTTP URL)
    message_base64 = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture?"),
            ImgContent(image_url=ImgUrl(url=base64_data_url)),
        ]
    )

    print("\n--- Analysing with base64 encoded image ---")
    response_base64 = client.get_completion(messages=[message_base64])
    print("\nModel response (base64):", response_base64.content)

    # ── Approach B: direct HTTP URL ──────────────────────────────────────────
    # Instead of embedding the image, we can pass a public HTTP URL.
    # DIAL Core (or the underlying model) fetches the image at inference time.
    # This keeps the request body small but requires the image to be publicly accessible.
    message_url = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture?"),
            ImgContent(image_url=ImgUrl(url="https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg")),
        ]
    )

    print("\n--- Analysing with direct HTTP URL ---")
    response_url = client.get_completion(messages=[message_url])
    print("\nModel response (URL):", response_url.content)


start()