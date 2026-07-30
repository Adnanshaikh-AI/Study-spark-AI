from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(uploaded_file):
    """
    Read PDF and return complete text.
    """
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text):
    """
    Split PDF into small chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    return docs