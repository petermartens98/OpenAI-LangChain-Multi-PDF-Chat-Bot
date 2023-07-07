import streamlit as st
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os

import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template
from codeDisplay import display_code
from textFunctions import get_pdf_text, get_pdfs_text, get_text_chunks
from vizFunctions import roberta_barchat, vaders_barchart
from prompts import set_prompt


def init_ses_states():
    session_states = {
        "conversation": None,
        "chat_history": None,
        "pdf_analytics_enabled": False,
        "display_char_count": False,
        "display_word_count": False,
        "display_vaders": False,
        "api_authenticated": False
    }
    for state, default_value in session_states.items():
        if state not in st.session_state:
            st.session_state[state] = default_value


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, temp, model):
    llm = ChatOpenAI(temperature=temp, model_name=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question, prompt):
    response = st.session_state.conversation({'question': (prompt+user_question)})
    st.session_state.chat_history = response['chat_history']
    with st.spinner('Generating response...'):
        display_convo(prompt)
        

def display_convo(prompt):
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message.content[len(prompt):]), unsafe_allow_html=True)


def process_docs(pdf_docs, TEMP, MODEL):
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    st.session_state["user_question"] = ""

    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True


def pdf_analytics(pdf_docs):
    all_text = ""
    if st.session_state.pdf_analytics_enabled:
        with st.expander("PDF Analytics", expanded=False):
            for pdf in pdf_docs:
                st.subheader(str(secure_filename(pdf.name)))
                text = get_pdf_text(pdf)
                all_text += text

                if st.session_state.display_word_count:
                    st.markdown(f'<p class="small-font"># of Words: {len(text.split())}</p>', unsafe_allow_html=True)

                if st.session_state.display_char_count:
                    st.markdown(f'<p class="small-font"># of Characters: {len(text)}</p>', unsafe_allow_html=True)

                if st.session_state.display_vaders:
                    vaders_barchart(text, name=str(secure_filename(pdf.name)))

            if len(pdf_docs) > 1:
                if any([st.session_state.display_word_count, st.session_state.display_char_count, st.session_state.display_vaders]):
                    st.subheader("Collective Summary:")
                    if st.session_state.display_word_count:
                        st.markdown(f'<p class="small-font"># of Words: {len(all_text.split())}</p>', unsafe_allow_html=True)

                    if st.session_state.display_char_count:
                        st.markdown(f'<p class="small-font"># of Characters: {len(all_text)}</p>', unsafe_allow_html=True)

                    if st.session_state.display_vaders:
                        vaders_barchart(all_text, name=str(secure_filename(pdf_docs[-1].name)))


class OpenAIAuthenticator:
    @staticmethod
    def authenticate(api_key):
        if not api_key: return False
        os.environ['OPENAI_API_KEY'] = api_key
        try:
            llm = OpenAI()
            if llm("hi"): return True
            else: return False
        except Exception:
            return False


def api_authentication():
    if not st.session_state.api_authenticated:
        openai_key = st.text_input("OpenAI API Key:", type="password")
        if not openai_key:
            st.info("Please enter your API Key.")
            return
        authenticator = OpenAIAuthenticator()
        if authenticator.authenticate(openai_key):
            st.session_state.api_authenticated = True
        elif not authenticator.authenticate(openai_key):
            st.session_state.api_authenticated = False
        
        if st.session_state.api_authenticated:
            st.success("Authentication Successful!")
        else:
            st.error("Invalid API Key. Please try again.")
    else:
        st.success("Authentication Successful!")


def chatbot_settings():
    global MODEL, PERSONALITY, TEMP
    with st.expander("Chat Bot Settings", expanded=True):
        MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo'])
        PERSONALITY = st.selectbox(label='Personality', options=['general assistant','academic','witty'])
        TEMP = st.slider("Temperature",0.0,1.0,0.5)


def pdf_analytics_settings():
    with st.expander("PDF Analytics Settings", expanded=True):
        enable_pdf_analytics = st.checkbox("Enable PDF Analytics")
        if enable_pdf_analytics:
            st.session_state.pdf_analytics_enabled = True
            st.caption("PDF Analytics Enabled")
            st.caption("Display Options")

            st.session_state.display_char_count = st.checkbox("Character Count")
            st.session_state.display_word_count = st.checkbox("Word Count")
            st.session_state.display_vaders = st.checkbox("VADER Sentiment Analysis")

        else:
            st.session_state.pdf_analytics_enabled = False
            st.caption("PDF Analytics Disabled")

def sidebar():
    global pdf_docs
    with st.sidebar:
        with st.expander("OpenAI API Authentication", expanded=True):
            api_authentication()
        chatbot_settings()
        pdf_analytics_settings()
        with st.expander("Your Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
            if st.button("Process Files + New Chat"):
                if pdf_docs:
                    with st.spinner("Processing"):
                        process_docs(pdf_docs, TEMP, MODEL)
                else: 
                    st.caption("Please Upload At Least 1 PDF")
                    st.session_state.pdf_processed = False


def main():
    st.set_page_config(page_title="Multi-Document Chat Bot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    global MODEL, PERSONALITY, TEMP, pdf_docs
    init_ses_states()
    deploy_tab, code_tab= st.tabs(["Deployment", "Code"])
    with deploy_tab:
        st.title(":books: Multi-Document ChatBot 🤖")
        st.subheader("Powered by OpenAI + LangChain + Streamlit")
        sidebar()
        if st.session_state.get("pdf_processed") and st.session_state.api_authenticated:
            prompt = set_prompt(PERSONALITY)
            pdf_analytics(pdf_docs)
            with st.form("user_input_form"):
                user_question = st.text_input("Ask a question about your documents:")
                send_button = st.form_submit_button("Send")
            if send_button and user_question:
                handle_userinput(user_question, prompt)
        if not st.session_state.get("pdf_processed"): 
            st.caption("Please Upload Atleast 1 PDF Before Proceeding")
        if not st.session_state.api_authenticated:
            st.caption("Please Authenticate OpenAI API Before Proceeding")
    with code_tab:
        display_code()


if __name__ == '__main__':
    main()

