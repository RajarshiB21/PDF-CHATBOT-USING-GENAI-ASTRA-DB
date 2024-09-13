import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio
from PyPDF2 import PdfReader

#Here we setup the api keys, replace these with your own keys that you get from the websites
ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""
OPENAI_API_KEY = ""

#Here we initialize the model and the embedding part using OpenAI 
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#PDF Query bot title, put an emoji here if you like
st.title("PDF Query App")

#Here we create an uploader to be able to upload files
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

#Now we create the sidebar where we upload the files, it has a spinner that runs while the text is being read, extracted and processed and stored in the database
if st.sidebar.button("Process PDF"):
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            pdfreader = PdfReader(uploaded_file)
            #Raw text is being extracted here
            raw_text = ''
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content
            
            #The database is being initialized here using cassio
            cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
            astra_vector_store = Cassandra(
                embedding=embedding,
                table_name="qa_mini_demo",
                session=None,
                keyspace=None,
            )
            #Text is being splitted into chunks of size 2000 and of overlap 200
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)
            #Adding the text to the database
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
            "You are an assistant and your job is to provide answers based on the context"
            "of the provided document. If you cannot find an answer since the answer is not in the given context, "
            "respond with 'I don't know'. Here is the context from the document:\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        formatted_prompt = prompt_template.format(question=query_text, context=context)

        with st.spinner("Getting answer..."):
            answer = astra_vector_index.query(formatted_prompt, llm=llm).strip()
            if not answer:
                answer = "I don't know"

        st.write(f"**Answer:** {answer}")
        st.write("**Source Documents:**")
        for doc, score in relevant_docs:
            st.write(f"    [{score:.4f}] \"{doc.page_content[:84]}...\"")
