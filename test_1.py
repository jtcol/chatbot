import os
import re
import streamlit as st
import openai
import PyPDF2
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Updated Import
from langchain_community.vectorstores import FAISS           # Updated Import
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI                  # Updated Import
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="SOPhia", page_icon="🤵‍♀️")
st.title("📄 SOP handling intelligent agent 🤵‍♀️")

st.markdown("""
### 🌟 **Welcome to SOPhia (SOP handling intelligent agent)🤵‍♀️!**

This application leverages **LangChain** and **OpenAI's** powerful language models to provide an interactive question-and-answer interface based on your uploaded documents.

#### **Key Features:**
- **📁 Upload Multiple Documents**: Support for PDF and TXT files.
- **🔍 Intelligent Search**: Quickly find relevant information within your documents.
- **📑 Detailed Sources**: Answers come with references to the specific document sections.
- **⚡ Fast and Efficient**: Optimized for quick processing and responses.

#### **Getting Started:**
1. **Upload Documents**:
   - Click on the sidebar's "Upload Documents" section.
   - Select and upload your PDF or TXT files.
2. **Process Documents**:
   - After uploading, click on "Start Upload and Processing".
   - Wait for the progress bar to complete the processing.
3. **Ask Questions**:
   - Once processing is done, the question input box will appear after app is rerun.
   - Enter your query and receive detailed answers.
4. **Review History**:
   - Access your query history in the "Query History" tab to revisit past interactions.

#### **Why Use This App?**
- **Efficiency**: Save time by extracting information without manually searching through documents.
- **Accuracy**: Get precise answers backed by the content of your documents.
- **Transparency**: Always know the source of the information provided.

#### **Note:**
- Ensure your `.env` file is properly configured with your OpenAI API key.
- Uploaded documents are securely stored and processed locally.

🔒 **Your data privacy is our priority!**
""")

# ------------------- Session State Initialization -------------------

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

if 'processing' not in st.session_state:
    st.session_state['processing'] = False

if 'documents_uploaded' not in st.session_state:
    st.session_state['documents_uploaded'] = False

# ------------------- Datastore Initialization -------------------

# Define the path for the SQLite database
DB_PATH = 'documents.db'

@st.cache_resource
def get_db_connection(db_path=DB_PATH):
    """
    Returns a singleton SQLite connection.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    # Ensure foreign keys are enforced
    conn.execute("PRAGMA foreign_keys = 1")
    # Create table if it doesn't exist
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT UNIQUE,
                source TEXT,
                section TEXT
            )
        ''')
    return conn

def add_document_to_db(conn, content, source, section):
    """
    Adds a document to the datastore if it doesn't already exist.
    Returns the document ID.
    """
    try:
        with conn:
            conn.execute(
                'INSERT INTO documents (content, source, section) VALUES (?, ?, ?)', 
                (content, source, section)
            )
            doc_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            return doc_id
    except sqlite3.IntegrityError:
        # Document already exists, fetch its ID
        with conn:
            result = conn.execute('SELECT id FROM documents WHERE content = ?', (content,)).fetchone()
            return result[0] if result else None

def get_document_by_id(conn, doc_id):
    """
    Retrieves a document's content and metadata by its ID.
    """
    with conn:
        cursor = conn.execute('SELECT content, source, section FROM documents WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        if result:
            content, source, section = result
            return Document(page_content=content, metadata={"source": source, "section": section})
    return None

# Initialize the datastore connection using cached resource
conn = get_db_connection()

# ------------------- FAISS Vector Store Initialization -------------------

# Define the path for the FAISS index
FAISS_INDEX_PATH = 'vector_store.faiss'

def init_vector_store(embedding_model, index_path=FAISS_INDEX_PATH):
    """
    Initialize the FAISS vector store. Loads existing index if available; otherwise, sets to None.
    """
    if os.path.exists(index_path):
        try:
            vectorstore = FAISS.load_local(
                index_path, 
                embedding_model, 
                allow_dangerous_deserialization=True  # Enable deserialization
            )
            st.sidebar.success("✅ Loaded existing FAISS vector store.")
            return vectorstore
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load FAISS index: {e}")
            return None
    else:
        st.sidebar.info("ℹ️ FAISS vector store will be created upon uploading documents.")
        return None

def save_vector_store(vectorstore, index_path=FAISS_INDEX_PATH):
    """
    Saves the FAISS vector store to disk.
    """
    try:
        vectorstore.save_local(index_path)
        st.sidebar.success("✅ FAISS vector store saved.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save FAISS vector store: {e}")

# Initialize the embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize the vector store
vectorstore = init_vector_store(embeddings)

# Assign to session state
st.session_state['vectorstore'] = vectorstore

# ------------------- Text Preprocessing -------------------

def preprocess_text(text):
    """
    Preprocesses the text by removing excessive whitespace and newlines.
    """
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# ------------------- Document Extraction Functions -------------------

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file with source metadata.
    """
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                # Tag by page number; modify as needed for actual sections
                section = f"Page {page_num + 1}"
                text += page_text + "\n"
            else:
                st.warning(f"⚠️ No text found on page {page_num + 1} of {file.name}.")
        return text
    except Exception as e:
        st.error(f"❌ Error extracting text from {file.name}: {e}")
        return ""

# ------------------- File Validation -------------------

def is_safe_file(file):
    """
    Validates the uploaded file type.
    """
    allowed_extensions = ["txt", "pdf"]
    return file.name.split('.')[-1].lower() in allowed_extensions

# ------------------- Background Document Processing -------------------

def process_documents(uploaded_files):
    """
    Processes uploaded documents: extracts text, inserts into DB, and updates FAISS vector store.
    """
    conn = get_db_connection()
    documents_to_add = []
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files, 1):
        try:
            if uploaded_file.type == "application/pdf":
                # Extract text from the PDF
                text = extract_text_from_pdf(uploaded_file)
                section = "Full Document"  # Modify based on actual sections
            else:
                # Read TXT file
                text = uploaded_file.getvalue().decode("utf-8")
                section = "Full Document"
            
            if text:
                # Preprocess the extracted text
                text = preprocess_text(text)
                
                # Check if the document already exists in the datastore
                doc_id = add_document_to_db(conn, text, uploaded_file.name, section)
                if doc_id:
                    # If the document is newly added, prepare it for vector store
                    existing_doc = get_document_by_id(conn, doc_id)
                    if existing_doc:
                        documents_to_add.append(existing_doc)
                else:
                    st.info(f"ℹ️ Document '{uploaded_file.name}' already exists in the datastore. Skipping addition.")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
        # Update progress
        progress = idx / total_files
        progress_bar.progress(progress)
    
    if documents_to_add:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents_to_add)
        
        # Update or create the FAISS vector store
        if st.session_state['vectorstore'] is None:
            # Create FAISS vector store from the new documents
            vectorstore = FAISS.from_documents(texts, embeddings)
            st.session_state['vectorstore'] = vectorstore
            st.sidebar.success("✅ FAISS vector store created and documents added.")
        else:
            # Add new documents to the existing vector store
            st.session_state['vectorstore'].add_documents(texts)
            st.sidebar.success("✅ New documents added to the existing FAISS vector store.")
        
        # Save the updated vector store
        save_vector_store(st.session_state['vectorstore'])
        
        st.success("🎉 New documents processed and added to the knowledge base successfully!")
    else:
        st.info("ℹ️ No new documents to add.")
    
    st.session_state['documents_uploaded'] = True

# ------------------- Q&A Functionality -------------------

if st.session_state['documents_uploaded'] and st.session_state.get('vectorstore'):
    # Define the custom prompt
    custom_prompt = """
    You are an expert assistant that provides highly structured and detailed step-by-step instructions based only on the provided documents (SOPs). Each step you provide must be clearly derived from the content and must cite the section and the source document from where it was taken. If the query cannot be answered from the documents, reply with: 'I do not know based on the provided SOPs.' and ask for clarification.
    1. You must directly perform all instructions with reference to the appropriate sections of the knowledge base
    2. You must only refer to sections of the knowledge base which is relevant to your task.
    3. You must always review your output to determine if the facts are consistent with the knowledge base
    4. Do not do math calculations and just cite the data as it is.
    5. Cite text in verbatim as far as possible
    6. In your output, retain the keywords and tone from the documents.
    7. If the output to the instructions cannot be derived from the knowledge base, strictly only reply “There is no relevant information, please only query about SOP related information”.
    Documents: {context}
    
    Question: {question}
    
    Provide your answer as a step-by-step guide, ensuring that every step cites a specific section and the corresponding source document of the SOP.
    """
    
    # Create a PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt
    )
    
    # Initialize the language model
    llm = OpenAI(temperature=0, openai_api_key=api_key, max_tokens=1000)
    
    # Create a QA chain using load_qa_chain with the custom prompt
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",  # Options: "stuff", "map_reduce", "refine"
        prompt=prompt_template
    )
    
    # Create a retriever from the vectorstore
    retriever = st.session_state['vectorstore'].as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Initialize the RetrievalQA chain with the custom QA chain
    qa = RetrievalQA(
        combine_documents_chain=qa_chain,
        retriever=retriever
    )
    
    # Create tabs for querying and history
    query_tab, history_tab = st.tabs(["Ask a Question", "Query History"])
    
    if "my_text" not in st.session_state:
        st.session_state.my_text = ""
    def submit():
        st.session_state.my.text = st.session_state.widget
        st.session_state.widget = ""

    with query_tab:
        st.subheader("🗨️ Ask a Question:")
        query = st.text_input("Enter your query:", placeholder="What information are you looking for?")
        search_button = st.button("Search")
        
        if search_button:
            if query.strip() == "":
                st.error("❌ Query cannot be empty.")
            else:
                # Optional: Implement query limits
                MAX_QUERIES = 50
                if len(st.session_state.query_history) >= MAX_QUERIES:
                    st.warning("⚠️ You have reached the maximum number of queries for this session.")
                else:
                    with st.spinner("🔍 Generating answer..."):
                        try:
                            # Run the QA chain with the provided query
                            answer = qa.run(query)
                            
                            st.subheader("💡 Answer:")
                            st.write(answer)
                            
                            # Append to history
                            st.session_state.query_history.append({"query": query, "answer": answer})
                            
                            # Retrieve relevant documents for source display
                            relevant_docs = retriever.get_relevant_documents(query)
                            if relevant_docs:
                                st.subheader("📄 Source Documents and Sections:")
                                for doc in relevant_docs:
                                    source = doc.metadata.get('source', 'Unknown Document')
                                    section = doc.metadata.get('section', 'Unknown Section')
                                    st.write(f"- **Document:** {source} | **Section:** {section}")
                            else:
                                st.markdown("hello")
                        except Exception as e:
                            st.error(f"❌ An error occurred while generating the answer: {e}")
    
    with history_tab:
        if st.session_state.query_history:
            st.subheader("🕒 Query History:")
            for idx, entry in enumerate(st.session_state.query_history, 1):
                st.markdown(f"**{idx}. Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}\n")
        else:
            st.info("ℹ️ No queries yet. Ask a question to see the history here.")
else:
    if st.session_state['processing']:
        st.info("🔄 Building the knowledge base. Please wait...")
    else:
        st.info("📝 Please upload documents via the sidebar to build the knowledge base and rerun the app to query")

# ------------------- Sidebar: Upload Documents and Reset Functionality -------------------

with st.sidebar.expander("📂 Upload Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload documents (TXT, PDF):",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state['processing']:
        if st.button("Start Upload and Processing"):
            with st.spinner("🔄 Processing uploaded documents..."):
                st.session_state['processing'] = True
                process_documents(uploaded_files)
                st.session_state['processing'] = False
                st.sidebar.success("✅ Document upload and processing completed.")
    
    if st.session_state['processing']:
        st.sidebar.info("🔄 Processing documents... Please wait.")

with st.sidebar.expander("⚙️ Settings", expanded=False):
    if st.session_state['vectorstore'] and st.session_state['query_history']:
        if st.button("🔄 Reset Knowledge Base and Query History"):
            st.session_state['vectorstore'] = None  # Re-initialize vectorstore to None
            st.session_state['query_history'] = []
            # Clear the FAISS index file
            if os.path.exists(FAISS_INDEX_PATH):
                try:
                    os.remove(FAISS_INDEX_PATH)
                    st.sidebar.success("✅ FAISS vector store file deleted.")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to delete FAISS vector store file: {e}")
            # Clear the datastore as well
            with get_db_connection() as conn:
                conn.execute('DELETE FROM documents')
            st.sidebar.success("✅ Knowledge base and query history have been reset.")

# ------------------- Close Datastore Connection -------------------

import atexit

def close_connection():
    conn = get_db_connection()
    conn.close()

atexit.register(close_connection)