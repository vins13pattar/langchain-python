"""
05_multimodal_messages.py
=========================
Demonstrates MULTIMODAL message content — sending images, files, and
other non-text data to models that support it.

Concepts covered:
  - Image input from URL
  - Image input from base64-encoded data
  - File (PDF) input
  - Standard content block format (cross-provider)
  - Provider-native format (OpenAI style)
  - Checking model profile for multimodal support
  - Multimodal output (image generation models)

Multimodal = the model can process AND return data other than text:
  - Images (JPEG, PNG, GIF, WebP, …)
  - Documents (PDF, …)
  - Audio (WAV, MP3, …) — on supported models
  - Video (MP4, …) — on supported models

Not all models support all modalities. Check model.profile before sending.
"""

import os
import base64
import httpx
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")  # GPT-4o supports vision

print("=" * 60)
print("Multimodal Messages Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. CHECK MODEL PROFILE FOR MULTIMODAL SUPPORT
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Check model profile ────────────────────────────────")
try:
    profile = model.profile
    print(f"\n  model:          {model.model_name}")
    print(f"  image_inputs:   {profile.get('image_inputs', 'unknown')}")
    print(f"  tool_calling:   {profile.get('tool_calling', 'unknown')}")
    print(f"  max_input_tokens: {profile.get('max_input_tokens', 'unknown')}")
except AttributeError:
    print("  (model.profile not available — continuing anyway)")


# ════════════════════════════════════════════════════════════════════
# 2. IMAGE FROM URL — standard content block format
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Image from URL (standard content block) ────────────")

# Standard LangChain format — works across OpenAI, Anthropic, Gemini
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

message_url = HumanMessage(content=[
    {"type": "text", "text": "Describe what you see in this image briefly."},
    {"type": "image", "url": image_url},      # ← standard content block
])

print(f"\n  Sending image URL: {image_url[:60]}…")
try:
    response = model.invoke([message_url])
    print(f"  Response: {response.content}")
except Exception as e:
    print(f"  (Skipped: {e})")


# ════════════════════════════════════════════════════════════════════
# 3. IMAGE FROM URL — provider-native format (OpenAI style)
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Image from URL (OpenAI native format) ──────────────")

# OpenAI-native format also works directly in LangChain
message_openai_native = HumanMessage(content=[
    {"type": "text", "text": "What colors are dominant in this image?"},
    {
        "type": "image_url",                  # ← OpenAI native type
        "image_url": {"url": image_url},
    },
])

print(f"\n  Using OpenAI-native 'image_url' block type")
try:
    response = model.invoke([message_openai_native])
    print(f"  Response: {response.content}")
except Exception as e:
    print(f"  (Skipped: {e})")


# ════════════════════════════════════════════════════════════════════
# 4. IMAGE FROM BASE64 DATA
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Image from base64 data ─────────────────────────────")

def url_to_base64(url: str) -> tuple[str, str]:
    """Download an image URL and return (base64_string, mime_type)."""
    response = httpx.get(url, follow_redirects=True, timeout=10)
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    b64 = base64.b64encode(response.content).decode("utf-8")
    return b64, content_type

print(f"\n  Downloading image to encode as base64…")
try:
    img_b64, mime_type = url_to_base64(image_url)

    message_b64 = HumanMessage(content=[
        {"type": "text",  "text": "Describe the image in one sentence."},
        {
            "type":      "image",
            "base64":    img_b64,            # ← raw base64 string
            "mime_type": mime_type,          # ← required for base64 data
        },
    ])

    print(f"  mime_type:    {mime_type}")
    print(f"  base64 length: {len(img_b64)} chars")

    response = model.invoke([message_b64])
    print(f"  Response: {response.content}")
except Exception as e:
    print(f"  (Skipped: {e})")


# ════════════════════════════════════════════════════════════════════
# 5. IMAGE FROM LOCAL FILE
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Image from local file ──────────────────────────────")

def image_file_to_message(path: str | Path) -> HumanMessage:
    """Create a HumanMessage from a local image file."""
    path = Path(path)
    suffix_to_mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    mime_type = suffix_to_mime.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return HumanMessage(content=[
        {"type": "text",  "text": "Describe this image in one sentence."},
        {"type": "image", "base64": b64, "mime_type": mime_type},
    ])

# Show the helper — won't run unless a local image exists
sample_path = Path("sample.png")
if sample_path.exists():
    msg = image_file_to_message(sample_path)
    response = model.invoke([msg])
    print(f"\n  Response: {response.content}")
else:
    print(f"\n  No local file at '{sample_path}' — showing the helper function:")
    print("""
  def image_file_to_message(path):
      b64 = base64.b64encode(path.read_bytes()).decode()
      return HumanMessage(content=[
          {"type": "text",  "text": "Describe this image."},
          {"type": "image", "base64": b64, "mime_type": "image/png"},
      ])
    """)


# ════════════════════════════════════════════════════════════════════
# 6. MULTI-IMAGE INPUT
#    Send several images in a single message
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Multiple images in one message ─────────────────────")

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Square_200x200.png/200px-Square_200x200.png",
]

multi_image_content = [{"type": "text", "text": "What do these two images have in common?"}]
for url in image_urls:
    multi_image_content.append({"type": "image", "url": url})

print(f"\n  Sending {len(image_urls)} images in one message…")
try:
    response = model.invoke([HumanMessage(content=multi_image_content)])
    print(f"  Response: {response.content}")
except Exception as e:
    print(f"  (Skipped: {e})")


# ════════════════════════════════════════════════════════════════════
# 7. CONTENT BLOCK FORMAT SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n── 7. Content block format summary ──────────────────────")
print("""
  ┌─────────────────────────────────────────────────────────┐
  │ Standard content block types (work across all providers)│
  ├──────────────────┬──────────────────────────────────────┤
  │ type             │ required extra fields                │
  ├──────────────────┼──────────────────────────────────────┤
  │ "text"           │ text (str)                           │
  │ "image"          │ url OR (base64 + mime_type)          │
  │ "audio"          │ url OR (base64 + mime_type)          │
  │ "video"          │ url OR (base64 + mime_type)          │
  │ "file"           │ url OR (base64 + mime_type)          │
  │ "text-plain"     │ text (str), optional mime_type       │
  │ "reasoning"      │ reasoning (str)                      │
  │ "tool_call"      │ name, args, id                       │
  └──────────────────┴──────────────────────────────────────┘

  All models also accept:
  - Provider-native formats (e.g. OpenAI's "image_url")
  - OpenAI chat completions dict format

  ⚠️  Not all models support all modalities.
      Check model.profile["image_inputs"] before sending images.
""")
