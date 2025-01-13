from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

tools = [
  {
      "type": "function",
      "function": {
          "name": "get_weather",
          "description": "Get the weather at a location",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string"}
              },
          },
      },
  },
    {
      "type": "function",
      "function": {
          "name": "get_temp",
          "description": "Get the temperature at a location",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string"}
              },
          },
      },
  }
]

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[{"role": "user", "content": "What is the weather and temperature in Paris?"}],
  tools=tools,
  tool_choice='auto',
  n=1
)

print(completion)

