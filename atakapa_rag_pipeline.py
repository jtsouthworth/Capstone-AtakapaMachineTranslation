import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class GrammarRAGPipeline:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding model. 
        Downloads the lightweight Hugging Face model to run locally on your machine.
        """
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None

    def ingest_grammar_book(self, file_path):
        """
        Loads the text, chunks it semantically, and embeds into the FAISS database.
        """
        print(f"Loading grammar text from {file_path}")

        # 1. Load raw text file
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text=f.read()

        # 2. Cleaning text file
        cleaned_text = re.sub(r'\n\s*\d+\s*\n', '\n', raw_text)
        compact_lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        compact_text = '\n'.join(compact_lines)
        doc = Document(page_content=compact_text, metadata={'source': file_path})

        # 3. Chunking the text file
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1200,
            chunk_overlap = 250,
            separators = [
                r"\n(?=\[)",       # PRIORITY 1: Split right before a bracketed phonetic example (e.g., [cakatkoˊpcĕn])
                r"\n(?=\d+\))",    # PRIORITY 2: Split right before a numbered example (e.g., 158) )
                r"\n(?=[A-Z])",    # PRIORITY 3: Split right before a capitalized English paragraph
                "\n",              # Fallback 1: Any newline
                " ",               # Fallback 2: Spaces
                ""                 # Fallback 3: Character level
            ],
            is_separator_regex=True
        )

        chunks = text_splitter.split_documents([doc])
        print(f"Split Grammar into {len(chunks)} chunks")

        # 4. Embedding
        print("Embedding chunks and building FAISS index...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("Vector database built successfully!")

    def save_database(self, folder_path='atakapa_grammar_faiss'):
        if self.vector_store:
            self.vector_store.save_local(folder_path)
            print(f"Database saved to /{folder_path}")

    def query_grammar(self, question, top_k=3):
        if not self.vector_store:
            print("Error: Vector store not loaded.")
            return

        print(f"\n--- Searching for: '{question}' ---")
        results = self.vector_store.similarity_search(question, k=top_k)
        
        for i, doc in enumerate(results):
            print(f"\n[Result {i+1}]")
            print(doc.page_content)