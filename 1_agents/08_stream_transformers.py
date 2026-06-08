from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
load_dotenv()
from langgraph.stream import StreamTransformer


# Tools

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def discount_calculator(price: float, discount: float) -> str:
    """Calculate the discounted price."""
    try:
        discounted_price = price * (1 - discount / 100)
        return f"The discounted price is: {discounted_price:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # This is a placeholder implementation. In a real application, you would call a weather API.
    if location.lower() == "bengaluru":
        return "The current weather in Bengaluru is sunny with a temperature of 30°C."
    else:
        return f"Weather information for {location} is not available."
    

# --- Stream Transformers ---

class HideToolCalls(StreamTransformer):
    """Drops tool-call events from the stream so users don't see them."""

    def init(self):
        return {}
    
    def process(self, event):
        if event.get('type') == "tool_call":
            return False
        return True
    

class MaskSensitiveInfo(StreamTransformer):
    """Masks sensitive prices in the streamed output.
    
    Key concepts:
    - before_builtins = True → runs before built-in transformers snapshot the text
    - process() → called for every stream event, return True to keep, False to drop
    - We mutate event data in-place to mask sensitive values
    """

    before_builtins = True

    SENSITIVE_VALUES = ["2,000", "1,500", "2000", "1500"]

    def init(self):
        return {}

    def process(self, event):
        # We only care about "messages" events (the streamed LLM output)
        if event.get("method") != "messages":
            return True

        # data is a tuple: (payload_dict, metadata_dict)
        payload = event["params"]["data"][0]

        # Mask the finished content block (contains the full assembled text)
        content = payload.get("content", {})
        if isinstance(content, dict) and "text" in content:
            content["text"] = self._mask(content["text"])

        return True

    def _mask(self, text):
        """Replace sensitive price values with ****"""
        for price in self.SENSITIVE_VALUES:
            text = text.replace(price, "****")
        return text


# --- Create the agent with the masking transformer ---

agent = create_agent(
    name="MyAgent",
    model="openai:gpt-5.5",
    tools=[calculate, discount_calculator, get_weather],
    transformers=[
        # lambda scope: HideToolCalls(scope),
        lambda scope: MaskSensitiveInfo(scope)
    ]
)

# --- Run the agent and print the masked output ---

stream = agent.stream_events({
    "messages": [
        SystemMessage(content="You are a helpful assistant that can perform calculations and apply discounts using available tools only."),
        HumanMessage(content="I am purchasing 2 tshirts that cost 2000rs each and 1 pair of jeans that costs 1500rs. What is the total price? What is the discounted price of the total price if I have a 10% discount? What is the weather like today in Bengaluru?"),
    ],
}, version="v3")

# Get the final output (waits for the agent to complete)
final_state = stream.output
final_message = final_state["messages"][-1]

# The transformer masks the "content-block-finish" event in the messages stream.
# For the final state output, we apply the same masking to the assembled content.
def mask_output(text):
    for price in MaskSensitiveInfo.SENSITIVE_VALUES:
        text = text.replace(price, "****")
    return text

if isinstance(final_message.content, list):
    for block in final_message.content:
        if isinstance(block, dict) and "text" in block:
            print(mask_output(block["text"]))
else:
    print(mask_output(final_message.content))
