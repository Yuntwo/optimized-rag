import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from ensemble import ensemble_retriever_from_mods
from full_chain import create_full_chain, ask_question
import json

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


# Get ensemble retriever for mods
@st.cache_resource
def get_retriever(openai_api_key=None):
    mods = load_and_chunk_json('data/mods22_23.json')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    return ensemble_retriever_from_mods(mods, embeddings=embeddings)


def get_chain(openai_api_key=None):
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
    chain = create_full_chain(ensemble_retriever,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def load_and_chunk_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    for module in data:
        # Combine each json object into a string and add metadata
        page_content = (
            f"Module Code: {module['moduleCode']}\n"
            f"Title: {module['title']}\n"
            f"Description: {module['description']}\n"
            f"Credits: {module['moduleCredit']}\n"
            f"Department: {module['department']}\n"
            f"Faculty: {module['faculty']}\n"
            f"Workload: {module.get('workload', 'N/A')}\n"
            f"Semester Data: {module.get('semesterData', 'N/A')}\n"
        )

        # Create a Document object for each module with metadata
        document = Document(
            page_content=page_content,
            metadata={
                "moduleCode": module['moduleCode'],
                "title": module['title'],
                "moduleCredit": module['moduleCredit'],
                "department": module['department'],
                "faculty": module['faculty']
            }
        )
        documents.append(document)

    return documents


def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key)
        st.subheader("Ask me questions about this week's meal plan")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()
