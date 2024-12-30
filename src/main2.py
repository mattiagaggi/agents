from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv


load_dotenv()



def get_word_length(word: str) -> int:
    """Return the length of a word."""
    return len(word)

# Wrap your Python function in a Tool
tools = [
    Tool(
        name="get_word_length",
        func=get_word_length,
        description="Returns the length of a word."
    )
]

# Instantiate an LLM (OpenAI Chat model)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Initialize an agent that knows how to use your tool
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # or "chat-zero-shot-react-description"
    verbose=True
)

# Now just run the agent
response = agent.run("How many letters in 'eudca'?")
print(response)
