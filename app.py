import streamlit as st
from utils.pdf_utils import load_pdf, split_text
from utils.ai import (create_vectorstore,ask_ai,summarize_pdf,explain_topic,generate_quiz)

st.set_page_config(
    page_title="StudySpark AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudySpark AI")
st.caption("Your Personal AI Study Assistant")

# ---------------- Sidebar ----------------

st.sidebar.title("📂 Upload Notes")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:

    with st.spinner("Reading PDF..."):

        pdf_text = load_pdf(uploaded_file)

        docs = split_text(pdf_text)

        vectorstore = create_vectorstore(docs)

        st.session_state.vectorstore = vectorstore

    st.sidebar.success("✅ PDF Loaded Successfully")

else:
    st.info("👈 Upload a PDF to begin.")
    st.stop()

# ---------------- Tabs ----------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Chat",
        "📝 Summary",
        "🧠 Explain",
        "❓ Quiz"
    ]
)
# ---------------- Chat ----------------

with tab1:

    st.subheader("💬 Chat with your Notes")

    question = st.text_input(
        "Ask anything from your PDF"
    )

    if st.button("Ask AI"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):

                answer = ask_ai(
                    st.session_state.vectorstore,
                    question
                )

            st.success(answer)


# ---------------- Summary ----------------

with tab2:

    st.subheader("📝 Smart Summary")

    if st.button("Generate Summary"):

        with st.spinner("Generating Summary..."):

            summary = summarize_pdf(
                st.session_state.vectorstore
            )

        st.success(summary)


# ---------------- Explain ----------------

with tab3:

    st.subheader("🧠 Explain Any Topic")

    topic = st.text_input(
        "Enter topic name"
    )

    if st.button("Explain"):

        if topic.strip() == "":
            st.warning("Please enter a topic.")
        else:

            with st.spinner("Explaining..."):

                explanation = explain_topic(
                    st.session_state.vectorstore,
                    topic
                )

            st.success(explanation)
            # ---------------- Quiz ----------------

with tab4:

    st.subheader("❓ AI Quiz Generator")

    if st.button("Generate Quiz"):

        with st.spinner("Generating Quiz..."):

            quiz = generate_quiz(
                st.session_state.vectorstore
            )

        st.success(quiz)


# ---------------- Footer ----------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <h3>📚 StudySpark AI</h3>
        <p>Your Personal AI Study Assistant</p>
        <p>Developed by <b>Adnan Shaikh</b></p>
    </div>
    """,
    unsafe_allow_html=True
)