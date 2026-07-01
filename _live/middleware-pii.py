from langchain.agents.middleware import PIIMiddleware
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# Redact all emails in user input
# agent = create_agent(
#     "openai:gpt-5.5",
#     middleware=[
#         PIIMiddleware("email", strategy="block"),
#     ],
# )

# result = agent.invoke({"messages": [{"role":"user", "content":"Create a draft email to example@example.com with subject 'Q4 Results'"}]})

# print("\n Result for PII middleware\n", result)


# Use different strategies for different PII types
# agent = create_agent(
#     "openai:gpt-5.5",
#     middleware=[
#         PIIMiddleware("email", strategy="mask"),
#         PIIMiddleware("credit_card", strategy="mask"),
#         PIIMiddleware("url", strategy="redact"),
#         PIIMiddleware("ip", strategy="hash"),
#     ],
# )

# result = agent.invoke({"messages":[{"role":"user", "content":"Create a draft email to example@microdegree.com with content 'Here is test credit card information Test Card no: 5555555555554444' and URL: https://www.amex.com and user IP address is: 192.168.0.1 "}]})

# print("\n Result after PII middleware\n", result)


# Custom PII type with regex
# agent = create_agent(
#     "openai:gpt-5.5",
#     middleware=[
#         PIIMiddleware("api_key", detector=r"\bsk-proj-[A-Za-z0-9_-]{20,}\b", strategy="block"),
#     ],
# )

# result = agent.invoke({"messages":[{"role":"user", "content":"My OpenAI api key is sk-proj-PLACEHOLDER_API_KEY_1234567890"}]})

# print("\n Result after PII middleware\n", result)


# agent = create_agent(
#     "openai:gpt-5.5",
#     middleware=[
#         PIIMiddleware("aadhaar", detector=detect_aadhaar, strategy="mask"),
#     ],
# )

# result = agent.invoke({"messages":[{"role":"user", "content":"My Aadhaar number is 2345 6789 1234."}]})

# print("\n Result after PII middleware\n", result)


# Custom PII type with function (Indian Aadhaar Number)
import re
from typing import List
from langchain.agents.middleware._redaction import PIIMatch

def _passes_verhoeff(aadhaar_number: str) -> bool:
    """Validate Aadhaar number using Verhoeff algorithm."""
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    
    c = 0
    clean_number = re.sub(r'[\s-]', '', aadhaar_number)
    if len(clean_number) != 12:
        return False
        
    for i, num in enumerate(reversed(clean_number)):
        c = d[c][p[i % 8][int(num)]]
    return c == 0

def detect_aadhaar(content: str) -> List[PIIMatch]:
    """Detect Indian Aadhaar numbers in content.
    
    Args:
        content: The text content to scan for Aadhaar numbers.
        
    Returns:
        A list of detected Aadhaar matches.
    """
    # Aadhaar is a 12 digit number, first digit cannot be 0 or 1.
    # Can be formatted as XXXX XXXX XXXX or XXXX-XXXX-XXXX or XXXXXXXXXXXX
    pattern = r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []
    
    for match in re.finditer(pattern, content):
        aadhaar_candidate = match.group()
        if _passes_verhoeff(aadhaar_candidate):
            matches.append(
                PIIMatch(
                    type="aadhaar",
                    value=aadhaar_candidate,
                    start=match.start(),
                    end=match.end(),
                )
            )
            
    return matches


agent = create_agent(
    model="openai:gpt-5.5",
    middleware=[
        PIIMiddleware("aadhaar", detector=detect_aadhaar, strategy="mask"),
    ],
)

result = agent.invoke({"messages":[{"role":"user", "content":"This is a test Aadhaar number is 2345 6789 1234. repeat it."}]})

print("\n Result after PII middleware\n", result)