from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Multiple PDFs
def get_pdfs_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        text += get_pdf_text(pdf)
    return text


# Single PDF
def get_pdf_text(pdf):
    text = ""
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

