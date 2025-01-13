from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "nullf",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
            },
        },
    }
]

# Create a chat completion request with streaming enabled.
response_stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "How many fingers on my hand?"}],
    tools=tools,
    tool_choice='auto',
    stream=True  # Enable streaming
)

# Stream the completion tokens or function call chunks as they arrive
for chunk in response_stream:
    # Each chunk will include partial data, often in the structure:
    # { "choices": [ { "delta": { "content": ... } } ] }
    # The exact structure may differ if using function calls.
    delta = chunk.choices[0].delta.content
    
    # If there's content, print it out as it streams
    
    print(delta, flush=True)

# Optional: Print a final newline after streaming finishes.
print()
