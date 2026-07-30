import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vectorstore(docs):
    """
    Convert PDF chunks into a searchable FAISS database.
    """
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


def ask_ai(vectorstore, question):
    """
    Ask questions using only relevant PDF chunks.
    """

    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are StudySpark AI.

Answer ONLY using the notes below.

If the answer is not available,
say:
"I couldn't find that in your notes."

NOTES:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
def summarize_pdf(vectorstore):
    """
    Generate a summary using the most important chunks.
    """

    docs = vectorstore.similarity_search(
        "Give a complete summary of this PDF",
        k=8
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Summarize these notes in simple bullet points.

NOTES:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def explain_topic(vectorstore, topic):
    """
    Explain a topic from the uploaded notes.
    """

    docs = vectorstore.similarity_search(
        topic,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Explain this topic in very simple words.

Topic:
{topic}

Notes:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_quiz(vectorstore):
    """
    Generate quiz questions from the uploaded notes.
    """

    docs = vectorstore.similarity_search(
        "important topics",
        k=8
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Create 10 MCQs from these notes.

For each question provide:

Question

A)

B)

C)

D)

Correct Answer

Notes:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content
