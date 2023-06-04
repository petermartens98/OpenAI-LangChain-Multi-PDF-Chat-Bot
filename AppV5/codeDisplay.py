import streamlit as st

def display_file_code(filename):
    with open(filename, "r") as file:
        code = file.read()
    with st.expander(filename, expanded=False):
        st.code(code, language='python')


def display_code():
    st.header("Source Code")
    display_file_code("main.py")
    display_file_code("htmlTemplates.py")
    display_file_code("textFunctions.py")
    display_file_code("vizFunctions.py")
    display_file_code("codeDisplay.py")
    display_file_code("prompts.py")
    display_file_code("requirements.txt")
