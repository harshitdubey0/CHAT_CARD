import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline


# =========================
# EMBEDDINGS
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# LOAD LLM
# =========================

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)


# =========================
# GLOBAL VECTORSTORE
# =========================

vectorstore = None


# =========================
# PROCESS PDF
# =========================

def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    return FAISS.from_documents(docs, embedding_model)


# =========================
# STRICT PROMPT
# =========================

prompt = ChatPromptTemplate.from_template(
    """
You are a strict question-answering assistant.
Use ONLY the context below to answer.
Give a short, precise answer (2-4 lines maximum).
Do NOT repeat the full paragraph.
If answer is not found, say "I don't know".
Context:
{context}
Question:
{question}
Answer:
"""
)


# =========================
# QA FUNCTION
# =========================

def answer_question(pdf, question):
    global vectorstore

    if pdf is not None:
        vectorstore = process_pdf(pdf)

    if vectorstore is None:
        return "Please upload a PDF first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
# =========================
# GRADIO UI (MODERN)
# =========================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
    ),
    css="""
    .main-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 30px;
        border-radius: 25px;
    }

    .chat-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .ask-btn {
        background: linear-gradient(to right, #2193b0, #6dd5ed);
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        height: 45px;
    }

    .ask-btn:hover {
        transform: scale(1.05);
        transition: 0.3s ease;
    }

    .status-text {
        font-weight: bold;
        color: #2c5364;
    }
    """
) as app:

    with gr.Column(elem_classes="main-container"):

        gr.Markdown("""
        # 🤖📄 Smart RAG PDF Assistant  
        ### Upload your PDF & ask anything from your document  

        ✅ Answers strictly from context  
        ✅ Short & precise responses  
        """)

        with gr.Column(elem_classes="chat-card"):

            pdf_input = gr.File(
                file_types=[".pdf"],
                label="📂 Upload PDF"
            )

            status_output = gr.Markdown("⚠️ No PDF uploaded", elem_classes="status-text")

            chatbot = gr.Chatbot(label="💬 Chat")

            question_input = gr.Textbox(
                placeholder="Ask something from your PDF...",
                show_label=False
            )

            ask_button = gr.Button("🚀 Ask", elem_classes="ask-btn")

        # =========================
        # HANDLERS
        # =========================

        def update_status(pdf):
            if pdf:
                return "✅ PDF Loaded Successfully!"
            return "⚠️ No PDF uploaded"

        pdf_input.change(update_status, inputs=pdf_input, outputs=status_output)

        def chat_interface(pdf, question, history):
            if not question:
                return history

            answer = answer_question(pdf, question)

            history.append((question, answer))
            return history

        ask_button.click(
            chat_interface,
            inputs=[pdf_input, question_input, chatbot],
            outputs=chatbot
        )

        question_input.submit(
            chat_interface,
            inputs=[pdf_input, question_input, chatbot],
            outputs=chatbot
        )

if __name__ == "__main__":
    app.launch()
