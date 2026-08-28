from chromadb.api.types import QueryResult
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from datetime import datetime
from uuid import uuid4
from langchain_core.documents import Document

import chromadb
import os

class Stor:
    def __init__(
        self,
        db_path: str = './chroma_data',
        model_name='all-MiniLM-L6-v2',
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        offline=False
    ):
        if offline:
            os.environ['HF_HUB_OFFLINE'] = "1"

        self.stor = chromadb.PersistentClient(path=db_path)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.collections = dict()
        self.model = SentenceTransformer(model_name, device='cpu')

    def append(self, collection_name: str, path_to_file: str):
        text = self.load_file(path_to_file)
        chunks = self.create_chunks(text)
        vec_chunks = self.vectorise_chunks(chunks)
        collection = self.get_collection(collection_name)
        date = datetime.now().date().isoformat()
        chunk_len = len(chunks)
        collection.add(
            documents=chunks,
            embeddings=vec_chunks,
            metadatas=[
                {
                    "source": path_to_file,
                    "date": date,
                    "chunk_index": i,
                    "chunk_count": chunk_len
                } for i in range(chunk_len)
            ],
            ids=[str(uuid4()) for _ in range(chunk_len)]
        )

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        if collection_name not in self.collections:
            self.collections[collection_name] = self.stor.get_or_create_collection(collection_name)
        return self.collections[collection_name]

    def list_collections(self):
        collections = self.stor.list_collections()
        for collection in collections:
            self.collections[collection.name] = collection
        return [collection.name for collection in collections]

    def text_query(self, collection_name: str, text: str, k: int = 3, returning_embeds=False) -> QueryResult:
        query = self.vectorise(text)
        collection = self.get_collection(collection_name=collection_name)
        resulting = collection.query(
            query_embeddings=[query],
            n_results=k,
            include=['documents', 'metadatas', 'embeddings', 'distances']
        ) if returning_embeds else collection.query(
            query_embeddings=[query],
            n_results=k
        )
        return resulting

    def vectorise(self, text: str):
        vec = self.model.encode(text)
        return vec

    def dump_collection(self, collection_name, returning_embeds=False):
        collection = self.get_collection(collection_name)
        return collection.get() if not returning_embeds else collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )

    def doc2text(self, docs: list[Document]):
        return [doc.page_content for doc in docs]

    def vectorise_chunks(self, chunks: list[Document]):
        vectors = self.model.encode(chunks)
        return vectors

    def load_file(self, path_to_file) -> list[Document]:
        loader = TextLoader(path_to_file)
        docs = loader.load()
        return docs

    def create_chunks(self, text) -> list[Document]:
        docs = self.splitter.split_documents(text)
        return self.doc2text(docs)