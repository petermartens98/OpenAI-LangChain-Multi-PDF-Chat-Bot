# Imports
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, temp, model):
    llm = ChatOpenAI(temperature=temp, model_name=model)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    with st.spinner('Generating response...'):
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize Session States
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("Chat with multiple PDFs :books:")
    st.subheader("Powered by OpenAI + LangChain + Streamlit")

    with st.sidebar.expander("Settings", expanded=True):
        MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
        TEMP = st.slider("Temperature",0.0,1.0,0.5)

    with st.sidebar.expander("Your Documents", expanded=True):
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

    st.sidebar.subheader("Click 'Process' to Initiate New Chat")
    if st.sidebar.button("Process"):
        if pdf_docs:
            with st.spinner("Processing"):
                # Cleaning up for new chat
                st.session_state["conversation"] = None
                st.session_state["chat_history"] = None
                st.session_state["user_question"] = ""

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)

                # Set the flag to indicate PDF processing is complete
                st.session_state.pdf_processed = True

        else: 
            st.sidebar.caption("Please Upload At Least 1 PDF")
            # Set the flag to indicate PDF processing is not complete
            st.session_state.pdf_processed = False

    if st.session_state.get("pdf_processed"):
        with st.form("user_input_form"):
            user_question = st.text_input("Ask a question about your documents:")
            send_button = st.form_submit_button("Send")
        if send_button and user_question:
            handle_userinput(user_question)
    else: 
        st.caption("Please Upload Atleast 1 PDF Before Proceeding")

if __name__ == '__main__':
    main()
