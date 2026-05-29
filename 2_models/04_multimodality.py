"""
04_multimodality.py
===================
Demonstrates MULTIMODALITY — passing non-text inputs (like images) to chat models.

Concepts covered:
  - Multimodal input format (list of content blocks in HumanMessage)
  - Image URLs inside requests
  - Local images using base64 encoding
  - Combining text instructions with image analysis

Many modern chat models (e.g. gpt-4o, gpt-4o-mini, claude-3-5-sonnet, gemini-2.0-flash)
are natively multimodal and can interpret visual inputs directly.
"""

import base64
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

print("=" * 60)
print("Model Multimodality Demo")
print("=" * 60)

# Initialize standard multimodal model
# (Ensure your chosen model supports vision inputs, e.g. gpt-4o-mini)
model = init_chat_model("openai:gpt-4o-mini")


# ════════════════════════════════════════════════════════════════════
# 1. ANALYZING AN IMAGE VIA PUBLIC URL
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("1. Image analysis via public URL")
print("─" * 60)

# We use a beautiful public placeholder image from picsum
IMAGE_URL = "https://picsum.photos/seed/antigravity/600/400"
print(f"Analyzing image URL: {IMAGE_URL}")

# Construct a HumanMessage where content is a LIST of dicts (content blocks)
# each block has a type: "text" or "image_url"
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe what you see in this image, including the main colors, composition, and emotional vibe."
        },
        {
            "type": "image_url",
            "image_url": {"url": IMAGE_URL}
        }
    ]
)

print("Sending multimodal request...")
try:
    response = model.invoke([message])
    print(f"\n🤖 Model Analysis:\n{response.content}")
except Exception as e:
    print(f"\n❌ Error during vision request: {e}")


# ════════════════════════════════════════════════════════════════════
# 2. LOCAL IMAGE VIA BASE64 ENCODING
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("2. Local image analysis using base64 encoding")
print("─" * 60)

# In a production app, you might have local files uploaded by users.
# Here is the utility pattern to base64 encode local images:

def encode_image(image_path: str) -> str:
    """Read a local file and convert it into a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Let's write the base64 structure demonstration without needing a physical file:
demo_base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" # tiny 1x1 black dot png

print("\nConstructing base64 image block...")
base64_message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What is the content or color of this image data?"
        },
        {
            "type": "image_url",
            "image_url": {
                # Format: data:<mime_type>;base64,<base64_string>
                "url": f"data:image/png;base64,{demo_base64_data}"
            }
        }
    ]
)

print("Sending base64 request...")
try:
    response2 = model.invoke([base64_message])
    print(f"\n🤖 Model Analysis of base64 data:\n{response2.content}")
except Exception as e:
    print(f"\n❌ Error during base64 request: {e}")


# ════════════════════════════════════════════════════════════════════
# 3. SCHEMA REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("3. Content Block Schema Reference")
print("─" * 60)
print("""
  For Multimodal LLM inputs:
  1. Wrap the message in a HumanMessage.
  2. Define 'content' as a list of dicts.
  3. Image URL Format:
     {
       "type": "image_url",
       "image_url": {
         "url": "https://example.com/image.jpg",
         "detail": "auto"   # Optional: "low", "high", or "auto"
       }
     }
  4. Local Image Format:
     {
       "type": "image_url",
       "image_url": {
         "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
         "detail": "high"
       }
     }
""")
