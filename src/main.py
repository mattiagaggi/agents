from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool


from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import tool

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import AgentExecutor



load_dotenv()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
result=agent_executor({"input": "How many letters in the word eudca"})
print("Intermediate Steps:")
for step in result["intermediate_steps"]:
    print(step)

print("\nFinal Output:")
print(result["output"])