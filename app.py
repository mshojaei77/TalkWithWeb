import os
import re
import uuid
import requests
import pickle
import streamlit as st
from bs4 import BeautifulSoup
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from web_scraper import scrape_with_depth_sync
import PyPDF2
import json

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Add session management to avoid reloading resources
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = get_openai_client()

# Initialize FAISS index and document store
embedding_size = 1536  # OpenAI embedding dimension
INDEX_FILE = "embeddings.faiss"
DOCUMENT_STORE_FILE = "document_store.pkl"

# Load existing index and document store if they exist
@st.cache_resource
def load_index_and_documents():
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCUMENT_STORE_FILE):
        with st.spinner("Loading knowledge base..."):
            index = faiss.read_index(INDEX_FILE)
            
            with open(DOCUMENT_STORE_FILE, 'rb') as f:
                document_store = pickle.load(f)
                
            return index, document_store
    else:
        index = faiss.IndexFlatL2(embedding_size)
        document_store = []
        return index, document_store

# Save index and document store to disk
def save_index_and_documents(index, document_store):
    with st.spinner("Saving knowledge base..."):
        faiss.write_index(index, INDEX_FILE)
        
        with open(DOCUMENT_STORE_FILE, 'wb') as f:
            pickle.dump(document_store, f)
        
        st.sidebar.success(f"✅ Saved {index.ntotal} vectors and {len(document_store)} documents")

# Extract URLs from text using regex
def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

# Scrape content from a URL and convert to text
def scrape_url(url):
    try:
        with st.spinner(f"Scraping {url}..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text content
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return ""

# Save content to a markdown file with website hash as filename
def save_to_markdown(content, url):
    url_hash = uuid.uuid4().hex[:8] if not url else hash(url) % 10000000
    filename = f"scraped_{url_hash}.md"
    
    # Check if file already exists
    if os.path.exists(filename):
        return filename
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Content from {url}\n\n")
        f.write(content)
    
    return filename

# Batch embedding function - much faster than one-by-one
@st.cache_data(ttl=3600)
def get_embeddings_batch(texts):
    with st.spinner("Generating text embeddings in batch..."):
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]

# Split content into chunks and add to FAISS index
def add_to_index(content, url, index, document_store):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    chunks = text_splitter.split_text(content)
    chunks_added = 0
    
    progress_bar = st.sidebar.progress(0)
    chunk_status = st.sidebar.empty()
    
    # Collect chunks to process in batch
    chunks_to_process = []
    chunk_sources = []
    
    for i, chunk in enumerate(chunks):
        # Update progress less frequently
        if i % 5 == 0 or i == len(chunks)-1:
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            chunk_status.text(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Check if this chunk from this URL is already in the document store
        chunk_exists = any(
            doc["text"] == chunk and doc["source"] == url 
            for doc in document_store
        )
        
        if not chunk_exists:
            chunks_to_process.append(chunk)
            chunk_sources.append(url)
    
    # Process embeddings in batch if there are chunks to process
    if chunks_to_process:
        # Get embeddings for all chunks at once
        embeddings = get_embeddings_batch(chunks_to_process)
        
        # Add embeddings to index and document store
        for i, (chunk, embedding) in enumerate(zip(chunks_to_process, embeddings)):
            # Convert to numpy array and reshape
            embedding_np = np.array(embedding).astype('float32').reshape(1, -1)
            
            # Add to FAISS index
            index.add(embedding_np)
            
            # Store document with metadata
            document_store.append({
                "text": chunk,
                "source": chunk_sources[i]
            })
            
            chunks_added += 1
    
    progress_bar.empty()
    chunk_status.empty()
    
    # Show success message if we added anything
    if chunks_added > 0:
        st.sidebar.success(f"Added {chunks_added} new chunks")
    
    # Delete markdown file after processing
    if url.startswith("PDF: "):
        # For PDFs, extract name and create consistent filename format
        file_name = url.replace("PDF: ", "")
        file_hash = hash(file_name) % 10000000
        md_filename = f"pdf_{file_hash}.md"
    else:
        # For URLs, use the same hash as when saving
        url_hash = hash(url) % 10000000 if url else uuid.uuid4().hex[:8]
        md_filename = f"scraped_{url_hash}.md"
    
    # Delete the file if it exists
    if os.path.exists(md_filename):
        try:
            os.remove(md_filename)
        except Exception as e:
            st.sidebar.warning(f"Could not remove temporary file {md_filename}: {e}")
        
    return chunks_added, index, document_store

# Search for relevant documents using FAISS
def search_documents(query, index, document_store, k=3):
    query_embedding = get_embeddings_batch([query])[0]  # Use the batch function
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Search in FAISS
    distances, indices = index.search(query_embedding_np, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(document_store):
            results.append({
                "text": document_store[idx]["text"],
                "source": document_store[idx]["source"],
                "score": float(distances[0][i])
            })
    
    return results

# Define a function to determine if RAG is needed for a query
def search_knowledge_base_tool(query, index, document_store, k=5):
    """
    Search the knowledge base for information relevant to the query.
    Only use this when the user asks for specific information that might be in the knowledge base.
    """
    results = search_documents(query, index, document_store, k=k)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    context = "Information from provided sources:\n\n"
    for i, result in enumerate(results, 1):
        context += f"[Source {i}: {result['source']}]\n{result['text']}\n\n"
    
    return context

# Process user input with function calling
def process_user_input(user_input, urls, index, document_store):
    # Define the function for RAG search
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for information relevant to the user's query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question to search for in the knowledge base."
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]
    
    # Process URLs if provided
    if urls:
        st.sidebar.info(f"Processing {len(urls)} URLs")
        
        url_progress = st.sidebar.empty()
        
        for i, url in enumerate(urls):
            url_progress.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            
            # Generate URL hash to check if already scraped
            url_hash = hash(url) % 10000000
            filename = f"scraped_{url_hash}.md"
            
            # Check if this URL is already in the index
            url_in_index = any(doc["source"] == url for doc in document_store)
            
            if url_in_index:
                st.sidebar.info(f"Content from {url} is already in the knowledge base")
                continue
            
            # Check if we've already scraped this URL
            if os.path.exists(filename):
                st.sidebar.info(f"Loading existing content for {url}")
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Add to FAISS index
                chunks_added, index, document_store = add_to_index(content, url, index, document_store)
            else:
                # Scrape URL if not already scraped
                content = scrape_url(url)
                if content:
                    # Save to markdown
                    filename = save_to_markdown(content, url)
                    st.sidebar.success(f"Saved content from {url}")
                    
                    # Add to FAISS index
                    chunks_added, index, document_store = add_to_index(content, url, index, document_store)
        
        url_progress.empty()
        
        # Save after processing all URLs
        save_index_and_documents(index, document_store)
    
    # If there's a question and we have RAG data available
    if user_input:
        # Detect if query is likely about documents in the knowledge base
        document_related_keywords = ["resume", "cv", "document", "pdf", "file", "report", "uploaded", "paper", 
                                    "summarize", "extract", "analyze", "read"]
        
        is_document_query = any(keyword in user_input.lower() for keyword in document_related_keywords)
        
        # Enhanced system prompt with clearer instructions
        system_message = (
            "You are a helpful assistant that answers questions based on the knowledge base. "
            "ALWAYS check the knowledge base first before responding to queries about specific documents or content. "
            "If the user asks about documents (like resumes, reports, etc.), ALWAYS search the knowledge base "
            "and only mention uploading if nothing relevant is found. "
            "If you find relevant content, summarize or analyze it as requested. "
            "Remember personal details the user shares with you like their name, preferences, or any information "
            "they provide during the conversation, and use this information in your responses when relevant."
        )
        
        # Initialize messages with system prompt
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history from session state
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                messages.append({"role": message["role"], "content": message["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        # For document queries, use tool_choice="required" to force RAG search
        if is_document_query and index.ntotal > 0:
            with st.spinner("Analyzing your question about documents..."):
                response = client.chat.completions.create(
                    model=st.session_state.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="required"  # Force the model to use the function for document queries
                )
        else:
            # For other queries, let the model decide
            with st.spinner("Processing your question..."):
                response = client.chat.completions.create(
                    model=st.session_state.model_name,
                    messages=messages,
                    tools=tools if index.ntotal > 0 else None,
                    tool_choice="auto"
                )
        
        assistant_message = response.choices[0].message
        
        # Check if the model called the function or was forced to
        if assistant_message.tool_calls:
            # Process the RAG search
            messages.append(assistant_message)
            
            results = []
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_knowledge_base":
                    query = function_args.get("query", user_input)
                    results = search_documents(query, index, document_store, k=st.session_state.search_k)
                    
                    # Build context from results
                    if results:
                        context = "Information from provided sources:\n\n"
                        for i, result in enumerate(results, 1):
                            context += f"[Source {i}: {result['source']}]\n{result['text']}\n\n"
                    else:
                        # Be explicit that nothing was found
                        if is_document_query:
                            context = "No relevant documents found in the knowledge base. Please advise the user to upload the document they're referring to."
                        else:
                            context = "No relevant information found in the knowledge base."
                    
                    # Add the function response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": context
                    })
            
            # Get final response with the RAG context
            with st.spinner("Generating response based on retrieved information..."):
                final_response = client.chat.completions.create(
                    model=st.session_state.model_name,
                    messages=messages,
                    temperature=st.session_state.temperature
                )
            
            return final_response.choices[0].message.content, results
        else:
            # Model decided not to use RAG, just return its response
            return assistant_message.content, []
    
    return None, []

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Main Streamlit app
def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize or load index and document store - only do this once per session
    if not st.session_state.initialized:
        index, document_store = load_index_and_documents()
        st.session_state.index = index
        st.session_state.document_store = document_store
        st.session_state.initialized = True
    else:
        index = st.session_state.index
        document_store = st.session_state.document_store
    
    # Display stats in sidebar
    with st.sidebar:
        st.title("Knowledge Base")
        st.metric("Vectors", index.ntotal)
        
        # Model selection dropdown
        st.markdown("---")
        st.subheader("Model Settings")
        if "model_name" not in st.session_state:
            st.session_state.model_name = "gpt-4o-mini"
        
        model_name = st.selectbox(
            "Select Model",
            options=["gpt-4o-mini", "gpt-4o", "o3-mini"],
            index=0,
            help="Choose which model to use for responses"
        )
        st.session_state.model_name = model_name
        
        # Show which URLs are in the knowledge base - optimize this to avoid recalculation
        if document_store and len(document_store) > 0:
            if "unique_sources" not in st.session_state:
                st.session_state.unique_sources = set(doc["source"] for doc in document_store)
            
            st.subheader("Sources in Knowledge Base:")
            for source in st.session_state.unique_sources:
                st.write(f"- {source}")
        
        # URL input section in sidebar
        st.subheader("Add Content to Knowledge Base")
        url_input = st.text_area("Enter URLs (one per line)", height=100, 
                                help="Enter URLs to websites you want to analyze")
        
        # Deep crawler option
        deep_crawler = st.checkbox("🕸️ Deep Crawler", help="Crawl links found on the given URLs")
        
        # Depth selector (only shown if deep crawler is enabled)
        max_depth = 1
        max_urls_per_level = 5
        if deep_crawler:
            col1, col2 = st.columns(2)
            with col1:
                max_depth = st.slider("Crawl Depth", min_value=1, max_value=3, value=1, 
                                     help="How many levels of links to follow (higher values will take longer)")
            with col2:
                max_urls_per_level = st.slider("URLs per Level", min_value=3, max_value=10, value=5,
                                              help="Maximum number of URLs to process at each depth level")
        
        urls = []
        if url_input:
            # Split by newline and extract URLs
            url_lines = url_input.split('\n')
            for line in url_lines:
                extracted = extract_urls(line)
                if extracted:
                    urls.extend(extracted)
                elif line.strip().startswith('http'):
                    # If the line itself looks like a URL
                    urls.append(line.strip())
        
        # Display the extracted URLs
        if urls:
            st.write("📋 URLs to process:")
            for url in urls:
                st.write(f"- {url}")
        
        process_urls = st.button("Process URLs", type="primary", disabled=len(urls) == 0)
        
        # Add a section for PDF uploads
        st.markdown("---")
        st.subheader("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files to add to knowledge base", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        process_pdfs = st.button("Process PDFs", type="primary", disabled=len(uploaded_files or []) == 0)
        
        # Add clear knowledge base button and functionality
        st.markdown("---")
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            with st.spinner("Clearing knowledge base and associated files..."):
                # Delete the files if they exist
                if os.path.exists(INDEX_FILE):
                    os.remove(INDEX_FILE)
                if os.path.exists(DOCUMENT_STORE_FILE):
                    os.remove(DOCUMENT_STORE_FILE)
                
                # Delete all markdown files
                for file in os.listdir():
                    if file.endswith(".md") or file.endswith(".pkl") or file.endswith(".faiss"):
                        os.remove(file)
                
                # Reset index and document store
                index = faiss.IndexFlatL2(embedding_size)
                document_store = []
                
                # Save empty index and document store
                save_index_and_documents(index, document_store)
                
                # Clear all Streamlit caches
                st.cache_resource.clear()
                st.cache_data.clear()
            
            st.success("Knowledge base and all associated files cleared successfully!")
            st.rerun()

        # Additional settings section
        st.markdown("---")
        st.subheader("Advanced Settings")
        
        # Number of search results to retrieve
        if "search_k" not in st.session_state:
            st.session_state.search_k = 5
        
        st.session_state.search_k = st.slider(
            "Number of Results to Retrieve (TOP K)", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.search_k,
            help="Number of relevant chunks to retrieve from the knowledge base"
        )
        
        # Text chunking settings
        col1, col2 = st.columns(2)
        with col1:
            if "chunk_size" not in st.session_state:
                st.session_state.chunk_size = 1000
            
            st.session_state.chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=2000, 
                value=st.session_state.chunk_size,
                step=100,
                help="Size of text chunks for embedding"
            )
        
        with col2:
            if "chunk_overlap" not in st.session_state:
                st.session_state.chunk_overlap = 100
            
            st.session_state.chunk_overlap = st.number_input(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=st.session_state.chunk_overlap,
                step=20,
                help="Overlap between text chunks"
            )
        
        # Temperature setting for model responses
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        
        st.session_state.temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness in responses (0=deterministic, 1=creative)"
        )
        
        # Toggle for debug mode
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        
        st.session_state.debug_mode = st.toggle(
            "Debug Mode", 
            value=st.session_state.debug_mode,
            help="Show additional information like search scores and processing details"
        )
            
        if process_urls and urls:
            if deep_crawler:
                # Use deep crawler to process URLs
                with st.spinner(f"Deep crawling URLs with depth {max_depth}..."):
                    st.info(f"This may take a while for depth {max_depth} with {max_urls_per_level} URLs per level")
                    results = scrape_with_depth_sync(urls, max_depth=max_depth, max_urls_per_level=max_urls_per_level)
                    
                    # Process each URL and its content from deep crawler results
                    total_processed = 0
                    crawler_progress = st.progress(0)
                    total_urls = len(results)
                    
                    for i, (url, data) in enumerate(results.items()):
                        crawler_progress.progress((i + 1) / total_urls)
                        st.sidebar.info(f"Processing {i+1}/{total_urls}: {url} (depth {data['depth']})")
                        
                        content = data['content']
                        if content:
                            # Save to markdown
                            filename = save_to_markdown(content, url)
                            
                            # Add to FAISS index
                            chunks_added, index, document_store = add_to_index(content, url, index, document_store)
                            total_processed += 1
                    
                    crawler_progress.empty()
                    save_index_and_documents(index, document_store)
                    st.success(f"Deep crawler processed {total_processed} URLs with content")
            else:
                # Process URLs without a question (regular mode)
                process_user_input("", urls, index, document_store)
                st.success("URLs processed and added to knowledge base!")
        
        if process_pdfs and uploaded_files:
            total_pdfs = len(uploaded_files)
            pdf_progress = st.progress(0)
            pdf_status = st.empty()
            
            for i, pdf_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / total_pdfs
                pdf_progress.progress(progress)
                pdf_status.text(f"Processing PDF {i+1}/{total_pdfs}: {pdf_file.name}")
                
                # Generate file hash to check if already processed
                file_hash = hash(pdf_file.name + str(pdf_file.size)) % 10000000
                filename = f"pdf_{file_hash}.md"
                
                # Check if this PDF is already in the knowledge base
                pdf_source = f"PDF: {pdf_file.name}"
                pdf_in_index = any(doc["source"] == pdf_source for doc in document_store)
                
                if pdf_in_index:
                    st.info(f"Content from {pdf_file.name} is already in the knowledge base")
                    continue
                
                # Extract text from PDF
                content = extract_text_from_pdf(pdf_file)
                
                if content:
                    # Save to markdown
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"# Content from PDF: {pdf_file.name}\n\n")
                        f.write(content)
                    
                    # Add to FAISS index with the optimized batch function
                    chunks_added, index, document_store = add_to_index(content, pdf_source, index, document_store)
                    # Update session state
                    st.session_state.index = index
                    st.session_state.document_store = document_store
                    # Update unique sources cache
                    st.session_state.unique_sources = set(doc["source"] for doc in document_store)
            
            pdf_progress.empty()
            pdf_status.empty()
            
            # Save after processing all PDFs
            save_index_and_documents(index, document_store)
            st.success(f"Processed {total_pdfs} PDF files and added to knowledge base!")

    st.title("Agentic RAG Assistant")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input at the bottom
    if question := st.chat_input("Message RAG-Powered AI Assistant..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get AI response
        with st.chat_message("assistant"):
            response, results = process_user_input(question, [], index, document_store)
            
            if response:
                st.markdown(response)
    
        # Add assistant response to chat history
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
