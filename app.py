import pandas as pd

import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent


def load_chain(df) -> ConversationChain:
    return create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)


st.set_page_config(page_title="LangPredict", page_icon=":robot:")
st.header("LangExplore")
st.write("Upload your dataset for GPT to explore")

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
