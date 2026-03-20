import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    Supported values depend on the model:
     - DALL-E 3:    '1024x1024', '1024x1792', '1792x1024'
     - gpt-image-1: '1024x1024', '1024x1536', '1536x1024', 'auto'
    """
    # DALL-E 3 values
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'
    # gpt-image-1 values
    square_gpt: str = '1024x1024'
    portrait_gpt: str = '1024x1536'
    landscape_gpt: str = '1536x1024'
    auto: str = 'auto'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
    Supported values depend on the model:
     - DALL-E 3:       'standard', 'hd'
     - gpt-image-1:    'low', 'medium', 'high', 'auto'
    """
    # DALL-E 3 values
    standard: str = "standard"
    hd: str = "hd"
    # gpt-image-1 values
    low: str = "low"
    medium: str = "medium"
    high: str = "high"
    auto: str = "auto"

async def _save_images(attachments: list[Attachment]):
    # DialBucketClient is an async context manager - it opens/closes the underlying
    # httpx.AsyncClient automatically, which is the recommended pattern for async HTTP.
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        for attachment in attachments:
            if not attachment.url:
                print(f"Skipping attachment '{attachment.title}' - no URL present")
                continue

            # The attachment URL is a relative path inside DIAL's file storage,
            # e.g. "files/<bucket_id>/generated_abc123.png".
            # get_file prepends "/v1/" to form the full API path before downloading.
            image_bytes = await bucket_client.get_file(url=attachment.url)

            # Use a timestamp in the filename so repeated runs don't overwrite each other.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"

            # Write the raw binary response to disk as a .png file.
            with open(filename, "wb") as f:
                f.write(image_bytes)

            print(f"Image saved locally as: {filename}")


def start() -> None:
    # DialModelClient targets an image generation model via DIAL.
    # The {model} placeholder in DIAL_CHAT_COMPLETIONS_ENDPOINT is replaced with
    # the deployment_name, giving e.g.:
    #   https://ai-proxy.lab.epam.com/openai/deployments/gpt-image-1-mini-2025-10-06/chat/completions
    #
    # Other available image generation models on this DIAL instance:
    #   "gpt-image-1.5-2025-12-16"            -> OpenAI GPT Image 1.5 (higher quality)
    #   "gemini-2.5-flash-image"               -> Google Gemini 2.5 Flash Image
    #   "gemini-3-pro-image-preview"           -> Google Gemini 3 Pro Image
    #   "gemini-3.1-flash-image-preview"       -> Google Gemini 3.1 Flash Image
    #   "stability.stable-image-core-v1:1"     -> StabilityAI Stable Image Core
    #   "stability.stable-image-ultra-v1:1"    -> StabilityAI Stable Image Ultra
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-image-1-mini-2025-10-06",
        api_key=API_KEY
    )

    # For image generation the message content IS the image prompt.
    # DALL-E interprets the text and produces a matching image.
    message = Message(
        role=Role.USER,
        content="Sunny day on Bali"
    )

    # custom_fields maps to the DIAL API's custom_fields.configuration object.
    # These parameters control the visual characteristics of the generated image:
    #
    #   size    - output resolution:
    #               "1024x1024" (square), "1024x1792" (portrait), "1792x1024" (landscape)
    #
    #   quality - rendering detail:
    #               "standard" (faster/cheaper) or "hd" (finer details, more consistent)
    #
    # Note: the "style" parameter ("vivid"/"natural") was specific to DALL-E 3 and is
    # not supported by gpt-image-1 or other models.
    #
    # DIAL API reference: https://dialx.ai/dial_api#operation/sendChatCompletionRequest
    response = client.get_completion(
        messages=[message],
        custom_fields={
            "size": Size.landscape_gpt,      # Wide landscape format (1536x1024, gpt-image-1 value)
            "quality": Quality.high,         # High-detail rendering pass (gpt-image-1 value)
        }
    )

    # DALL-E does not return the image inline; instead DIAL stores it in the bucket
    # and puts the download URL into custom_content.attachments of the response message.
    if response.custom_content and response.custom_content.attachments:
        # asyncio.run() bridges synchronous start() with the async _save_images().
        # It spins up a new event loop, runs the coroutine, then tears the loop down.
        asyncio.run(_save_images(response.custom_content.attachments))
    else:
        print("No image attachments found in the response.")
        print("Response content:", response.content)


start()
