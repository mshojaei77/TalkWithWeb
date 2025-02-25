import os
import re
import uuid
import requests
import pickle
from bs4 import BeautifulSoup
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from web_scraper import scrape_urls_sync

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS index and document store
embedding_size = 1536  # OpenAI embedding dimension
index = None
document_store = []

# File paths for saved data
INDEX_FILE = "embeddings.faiss"
DOCUMENT_STORE_FILE = "document_store.pkl"

# Load existing index and document store if they exist
def load_index_and_documents():
    global index, document_store
    
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCUMENT_STORE_FILE):
        print("Loading existing index and document store...")
        index = faiss.read_index(INDEX_FILE)
        
        with open(DOCUMENT_STORE_FILE, 'rb') as f:
            document_store = pickle.load(f)
            
        print(f"Loaded index with {index.ntotal} vectors and {len(document_store)} documents")
    else:
        print("Creating new index and document store...")
        index = faiss.IndexFlatL2(embedding_size)
        document_store = []

# Save index and document store to disk
def save_index_and_documents():
    print("Saving index and document store...")
    faiss.write_index(index, INDEX_FILE)
    
    with open(DOCUMENT_STORE_FILE, 'wb') as f:
        pickle.dump(document_store, f)
    
    print(f"Saved index with {index.ntotal} vectors and {len(document_store)} documents")

# Load existing data at startup
load_index_and_documents()

def extract_urls(text):
    """Extract URLs from text using regex."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

def scrape_url(url):
    """Scrape content from a URL and convert to markdown."""
    try:
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
        print(f"Error scraping {url}: {e}")
        return ""

def save_to_markdown(content, url):
    """Save content to a markdown file with website hash as filename."""
    # Create a hash of the URL to use as identifier
    url_hash = uuid.uuid4().hex[:8] if not url else hash(url) % 10000000
    filename = f"scraped_{url_hash}.md"
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"File for {url} already exists as {filename}")
        return filename
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Content from {url}\n\n")
        f.write(content)
    
    return filename

def get_embedding(text):
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def add_to_index(content, url):
    """Split content into chunks and add to FAISS index."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    chunks = text_splitter.split_text(content)
    chunks_added = 0
    
    for chunk in chunks:
        # Check if this chunk from this URL is already in the document store
        chunk_exists = any(
            doc["text"] == chunk and doc["source"] == url 
            for doc in document_store
        )
        
        if not chunk_exists:
            embedding = get_embedding(chunk)
            # Convert to numpy array and reshape
            embedding_np = np.array(embedding).astype('float32').reshape(1, -1)
            
            # Add to FAISS index
            index.add(embedding_np)
            
            # Store document with metadata
            document_store.append({
                "text": chunk,
                "source": url
            })
            
            chunks_added += 1
    
    # Save updated index and document store if we added anything
    if chunks_added > 0:
        save_index_and_documents()
        
    return chunks_added

def search_documents(query, k=3):
    """Search for relevant documents using FAISS."""
    query_embedding = get_embedding(query)
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

def process_user_input(user_input, urls=None):
    """
    Process user input, handle URLs, and generate response.
    
    Args:
        user_input (str): The user's query or message
        urls (list, optional): List of URLs to process. If None, URLs will be extracted from user_input
    """
    # Extract URLs from user input if not provided
    if urls is None:
        urls = extract_urls(user_input)
    
    context = ""
    if urls:
        print(f"Found {len(urls)} URLs to process.")
        
        # Remove URLs from user input if they were extracted from it
        if urls == extract_urls(user_input):
            for url in urls:
                user_input = user_input.replace(url, "").strip()
            
        for url in urls:
            # Generate URL hash to check if already scraped
            url_hash = hash(url) % 10000000
            filename = f"scraped_{url_hash}.md"
            
            # Check if this URL is already in the index
            url_in_index = any(doc["source"] == url for doc in document_store)
            
            if url_in_index:
                print(f"Content from {url} is already in the index. Skipping...")
                continue
            
            # Check if we've already scraped this URL
            if os.path.exists(filename):
                print(f"Using existing content for {url} from {filename}")
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Add to FAISS index
                chunks_added = add_to_index(content, url)
                print(f"Added {chunks_added} chunks to the index from existing file")
            else:
                # Scrape URL if not already scraped
                content = scrape_url(url)
                if content:
                    # Save to markdown
                    filename = save_to_markdown(content, url)
                    print(f"Saved content from {url} to {filename}")
                    
                    # Add to FAISS index
                    chunks_added = add_to_index(content, url)
                    print(f"Added {chunks_added} chunks to the index")
    
    # If we have data in the index and there's a question
    if index.ntotal > 0 and user_input:
        # Search for relevant documents
        results = search_documents(user_input, k=5)  # Increased k for better context
        
        # Create context from results
        if results:
            context = "Information from provided URLs:\n\n"
            for i, result in enumerate(results, 1):
                context += f"[Source {i}: {result['source']}]\n{result['text']}\n\n"
            
            context += "Please answer based on the above information."
    
    # Generate response using OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. "
                                    "When referencing information, mention which source it came from."}
    ]
    
    if context:
        messages.append({"role": "system", "content": context})
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message.content

def main():
    # Example 1: Extract URLs from user input
    user_input = "what is SpaceX backup launch dates base on https://spaceexplored.com/2025/02/25/starship-flight-8-spacex-moves-and-stacks-booster-15-overnight/"
    response = process_user_input(user_input)
    print(f"\nAssistant: {response}")
    
    # Example 2: Directly pass URLs
    # user_input = "what is SpaceX backup launch dates?"
    # urls = ["https://spaceexplored.com/2025/02/25/starship-flight-8-spacex-moves-and-stacks-booster-15-overnight/"]
    # response = process_user_input(user_input, urls)
    # print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
