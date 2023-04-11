import pandas as pd

import streamlit as st
from streamlit_chat import message

from langchain import LLMChain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory


PREFIX = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
SUFFIX = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


st.set_page_config(page_title="Auto-Explore", page_icon=":robot:")
st.header("Auto-Explore")
st.write("Upload your dataset for GPT to explore")


def load_chain(df) -> ConversationChain:
    dataframe_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    tools = [
        Tool(name="dataframe",
             func=dataframe_agent.run,
             description="dataframe manipulations")
    ]
    prompt = ZeroShotAgent.create_prompt(tools,
                                         prefix=PREFIX,
                                         suffix=SUFFIX,
                                         input_variables=["input", "chat_history", "agent_scratchpad"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_uploaded_file():
    text = "Start the conversation by uploading data in a CSV file"
    return st.file_uploader(text, type={"csv"})


def get_text():
    input_text = st.text_input("You: ", "How many rows are in this dataset?", key="input")
    return input_text


uploaded_file = get_uploaded_file()

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    chain = load_chain(data)

    st.write(uploaded_file)

    user_input = get_text()
    output = chain.run(input=user_input)  # input with data
    st.session_state.past.append(user_input)  # input with data
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
