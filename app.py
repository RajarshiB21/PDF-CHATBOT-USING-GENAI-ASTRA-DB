import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio
from PyPDF2 import PdfReader

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:lgvFFEbTNKTmXmknriQMvjdR:66000d773f7fd4eefba8c0e9c5b7e61b0aa2aa399188b34bbd8e9fc5962ee1ee"
ASTRA_DB_ID = "7d990bfc-85e4-41fc-b741-c71e46e089ff"
OPENAI_API_KEY = "sk-WjPkOp7j57utWsdNJSrK5EJJhziXY8eIYtRTEqID8IT3BlbkFJm3Suq9pky-YXTdOM352gQG2vt5Q_Sfah97SHS_e0sA"

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

st.title("PDF Query App")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if st.sidebar.button("Process PDF"):
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            pdfreader = PdfReader(uploaded_file)
            raw_text = ''
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content

            cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
            astra_vector_store = Cassandra(
                embedding=embedding,
                table_name="qa_mini_demo",
                session=None,
                keyspace=None,
            )

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)
            astra_vector_store.add_texts(texts)
            st.success(f"Inserted {len(texts)} headlines into the database.")

            astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

            st.session_state['index'] = astra_vector_index
            st.session_state['vector_store'] = astra_vector_store
    else:
        st.error("Please upload a PDF file first.")

if 'index' in st.session_state:
    query_text = st.text_input("Enter your question:")
    if query_text:
        astra_vector_store = st.session_state['vector_store']
        astra_vector_index = st.session_state['index']

        relevant_docs = astra_vector_store.similarity_search_with_score(query_text, k=4)
        context = "\n".join([doc.page_content[:84] for doc, score in relevant_docs])

        prompt_template = (
            "You are a helpful assistant that provides answers based only on the contents "
            "of the provided document. If the answer to the question is not found in the document, "
            "respond with 'I don't know'. Here is the context from the document:\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        formatted_prompt = prompt_template.format(question=query_text, context=context)

        with st.spinner("Fetching answer..."):
            answer = astra_vector_index.query(formatted_prompt, llm=llm).strip()
            if not answer:
                answer = "I don't know"

        st.write(f"**Answer:** {answer}")
        st.write("**Referenced Documents:**")
        for doc, score in relevant_docs:
            st.write(f"    [{score:.4f}] \"{doc.page_content[:84]}...\"")
