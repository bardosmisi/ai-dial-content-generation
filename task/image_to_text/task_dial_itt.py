import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'

    # DialBucketClient is an async context manager.
    # "async with" opens the underlying httpx.AsyncClient on entry and closes it on exit,
    # which correctly releases the network connection even if an error is raised.
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:

        # Read the image as raw bytes, then wrap in BytesIO.
        # httpx's file upload API requires a file-like object (read() interface),
        # which BytesIO provides without needing a temporary file on disk.
        with open(image_path, "rb") as image_file:
            image_buffer = BytesIO(image_file.read())

        # put_file internally calls GET /v1/bucket to retrieve the private bucket ID
        # for our API key, then uploads to PUT /v1/files/<bucket_id>/<name>.
        # The response JSON contains the relative URL of the stored file, e.g.:
        #   {"url": "files/<bucket_id>/dialx-banner.png", ...}
        result = await bucket_client.put_file(
            name=file_name,
            mime_type=mime_type_png,
            content=image_buffer
        )

        # "url" is a relative path inside DIAL's file storage.
        # Storing it in an Attachment lets DIAL Core resolve and forward the file
        # to whatever model vendor we choose - no re-uploading needed per request.
        url = result.get("url")

        return Attachment(
            title=file_name,    # Human-readable label (shown in the DIAL chat UI)
            url=url,            # Relative bucket path - DIAL Core uses this to fetch the file
            type=mime_type_png  # MIME type so the model knows how to interpret the bytes
        )


def start() -> None:
    # asyncio.run() is the standard way to execute a coroutine from synchronous code.
    # It creates a new event loop, runs _put_image() to completion, then closes the loop.
    attachment = asyncio.run(_put_image())

    # Print the attachment so we can verify the upload and see the bucket URL.
    print(f"Uploaded attachment: {attachment}")

    # Create a DialModelClient for a vision-capable model.
    # The key advantage of the DIAL bucket approach over base64 embedding:
    #   - The image is uploaded once and referenced by URL for any number of calls.
    #   - DIAL Core adapts the attachment format for each vendor automatically.
    # Try swapping the deployment_name to compare responses from different models:
    #   "gpt-4o"               -> OpenAI
    #   "claude-3-sonnet"      -> Anthropic
    #   "gemini-pro-vision"    -> Google
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # Build the user message.
    # - content       : the text question to ask about the image
    # - custom_content: wraps the list of attachments so DIAL Core knows which
    #                   files to include when it forwards the request to the model
    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )

    response = client.get_completion(messages=[message])
    print("\nModel response:", response.content)


start()
