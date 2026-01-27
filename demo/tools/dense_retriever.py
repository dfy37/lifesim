import json
import os
import datetime
import numpy as np
from tqdm.auto import tqdm

from typing import Any, List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

import chromadb
from chromadb.config import Settings
# from chromadb.utils import embedding_functions
from tools.embedding_func import MTSentenceTransformerEmbeddingFunction

from utils.utils import get_logger

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

class DenseRetriever:
    """
    Use ChromaDB as the vector database retrieval tool,
    supporting single queries and batch index construction without the need for external services.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "retriever_collection",
        max_length: int = 512,
        embedding_dim: int = 384,
        device: str = "auto",
        persist_directory: str = "./chroma_db",
        distance_function: str = "cosine",  # cosine, l2, ip
        use_custom_embeddings: bool = False,
        logger_silent: bool = False
    ):
        """
        Initialize the retriever

        Args:
            model_name: Name of the pre-trained model
            collection_name: Name of the ChromaDB collection
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
            device: Device type, "auto" for automatic selection
            persist_directory: Directory for data persistence
            distance_function: Distance function (cosine, l2, ip)
            use_custom_embeddings: Whether to use a custom embedding model
            logger_silent: whether to cancel logger usage
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.persist_directory = persist_directory
        self.distance_function = distance_function
        self.use_custom_embeddings = use_custom_embeddings
        self.logger = get_logger(__name__, silent=logger_silent)
        
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if self.use_custom_embeddings:
            self._load_model()

        self._init_chromadb()
    
    def _load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(torch.device(self.device))
            self.model.eval()
            if 'Qwen3' in self.model_name:
                self.pool_func = last_token_pool  # Qwen3 model use the last token
            else:
                self.pool_func = mean_pooling
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _init_chromadb(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            if self.use_custom_embeddings:
                embedding_function = None
            else:
                embedding_function = MTSentenceTransformerEmbeddingFunction(
                    model_name=self.model_name,
                    device = self.device
                )
            
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
                self.logger.info(f"Loaded existing collection '{self.collection_name}'")
            except Exception as e:
                collection_metadata = {"hnsw:space": self.distance_function}
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,
                    metadata=collection_metadata
                )
                self.logger.info(f"Created new collection '{self.collection_name}'")
            
            self.logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encoding texts to vectors
        """
        if not self.use_custom_embeddings:
            return None
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(torch.device(self.device))
        attention_mask = encoded['attention_mask'].to(torch.device(self.device))
        
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.pool_func(model_output.last_hidden_state, attention_mask)
            # L2 normalization
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def is_collection_empty(self) -> bool:
        """
        Determine whether the current collection was loaded from an existing persistent directory
        """
        try:
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                if self.collection.count() > 0:
                    return False
                else:
                    return True
            else:
                return True
        except Exception as e:
            return True


    def build_index(
        self, 
        data: List[Dict[str, Any]], 
        text_field: str = "text",
        id_field: str = "id",
        batch_size: int = 32,
        clear_existing: bool = True
    ):
        """
        Build the index from the provided data
        
        Args:
            data: List of data, each element is a dictionary
            text_field: Name of the text field
            id_field: Name of the ID field; if not provided, IDs will be generated automatically
            batch_size: Batch processing size
            clear_existing: Whether to clear existing data
        """
        self.logger.info(f"Building index for {len(data)} documents...")
        start_time = datetime.datetime.now()
        
        if clear_existing:
            try:
                existing_count = self.collection.count()
                if existing_count > 0:
                    self.logger.info(f"Deleting {existing_count} existing documents...")
                    # ChromaDB does not have a direct clear method; we need to delete the collection and recreate it.
                    self.client.delete_collection(self.collection_name)
                    collection_metadata = {"hnsw:space": self.distance_function}
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.collection._embedding_function,
                        metadata=collection_metadata
                    )
            except Exception as e:
                self.logger.warning(f"Failed to clear existing data: {e}")
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing documents"):
            batch_data = data[i:i+batch_size]
            
            batch_ids = []
            batch_texts = []
            batch_metadatas = []
            batch_embeddings = None
            
            for j, item in enumerate(batch_data):
                if text_field not in item:
                    raise ValueError(f"Text field '{text_field}' not found in data item {i+j}")
                text = str(item[text_field])
                batch_texts.append(text)
                
                if id_field in item:
                    doc_id = str(item[id_field])
                else:
                    doc_id = str(i + j)
                batch_ids.append(doc_id)
                
                metadata = {k: v for k, v in item.items() if k not in [id_field]}
                clean_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
                batch_metadatas.append(clean_metadata)
            
            if self.use_custom_embeddings:
                embeddings = self._encode_text(batch_texts)
                batch_embeddings = embeddings.tolist() if embeddings is not None else None
            
            try:
                if batch_embeddings is not None:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
                else:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
            except Exception as e:
                self.logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                raise
        
        end_time = datetime.datetime.now()
        total_docs = self.collection.count()
        self.logger.info(f"Index built successfully. Total docs: {total_docs}, Time: {(end_time-start_time).seconds}s")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        return_scores: bool = True,
        return_embeddings: bool = False,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar documents

        Args:
            query: Query text
            top_k: Return the top-k results
            return_scores: Whether to return similarity scores
            return_embeddings: Whether to return the query embedding vector
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            A list of search results, each containing the original data and optional scores
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        try:
            query_params = {
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if where is not None:
                query_params["where"] = where
            if where_document is not None:
                query_params["where_document"] = where_document
            
            if self.use_custom_embeddings:
                query_embedding = self._encode_text([query])
                if query_embedding is not None:
                    query_params["query_embeddings"] = query_embedding.tolist()
                    search_results = self.collection.query(**query_params)
                else:
                    query_params["query_texts"] = [query]
                    search_results = self.collection.query(**query_params)
            else:
                query_params["query_texts"] = [query]
                search_results = self.collection.query(**query_params)
            
            results = []
            ids = search_results.get('ids', [[]])[0]
            documents = search_results.get('documents', [[]])[0]
            metadatas = search_results.get('metadatas', [[]])[0]
            distances = search_results.get('distances', [[]])[0]
            
            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                data = {
                    "text": document,
                    **(metadata or {})
                }
                
                result = {
                    'rank': i + 1,
                    'id': doc_id,
                    'data': data
                }
                
                if return_scores:
                    # ChromaDB returns distances, which need to be converted into similarity scores.
                    if self.distance_function == "cosine":
                        result['score'] = 1.0 - float(distance)
                    elif self.distance_function == "l2":
                        result['score'] = 1.0 / (1.0 + float(distance))
                    else:
                        result['score'] = float(distance)
                
                if return_embeddings and self.use_custom_embeddings:
                    query_embedding = self._encode_text([query])
                    if query_embedding is not None:
                        result['query_embedding'] = query_embedding[0].tolist()
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            total_docs = self.collection.count()
            
            stats = {
                "status": "ready",
                "total_docs": total_docs,
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "device": str(self.device),
                "collection_name": self.collection_name,
                "distance_function": self.distance_function,
                "persist_directory": self.persist_directory,
                "use_custom_embeddings": self.use_custom_embeddings
            }
            
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def add_documents(
        self, 
        new_data: List[Dict[str, Any]], 
        text_field: str = "text",
        id_field: str = "id",
        batch_size: int = 32
    ):
        """
        Add new documents to the existing index

        Args:
            new_data: List of new documents
            text_field: Name of the text field
            id_field: Name of the ID field
            batch_size: Batch processing size
        """
        self.logger.info(f"Adding {len(new_data)} new documents to collection...")
        
        for i in tqdm(range(0, len(new_data), batch_size), desc="Adding new documents"):
            batch_data = new_data[i:i+batch_size]
            
            batch_ids = []
            batch_texts = []
            batch_metadatas = []
            batch_embeddings = None
            
            for j, item in enumerate(batch_data):
                if text_field not in item:
                    raise ValueError(f"Text field '{text_field}' not found in data item {i+j}")
                
                text = str(item[text_field])
                batch_texts.append(text)
                
                if id_field in item:
                    doc_id = str(item[id_field])
                else:
                    doc_id = f"auto_{datetime.datetime.now().timestamp()}_{i+j}"
                batch_ids.append(doc_id)
                
                metadata = {k: v for k, v in item.items() if k not in [text_field, id_field]}
                clean_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
                batch_metadatas.append(clean_metadata)
            
            if self.use_custom_embeddings:
                embeddings = self._encode_text(batch_texts)
                batch_embeddings = embeddings.tolist() if embeddings is not None else None
            
            try:
                if batch_embeddings is not None:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
                else:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
            except Exception as e:
                self.logger.error(f"Failed to add documents: {e}")
                raise
        
        total_docs = self.collection.count()
        self.logger.info(f"Successfully added documents. Total documents: {total_docs}")
    
    def delete_documents(self, doc_ids: List[str]):
        """
        Delete documents with specified IDs

        Args:
            doc_ids: List of document IDs to be deleted
        """
        if not doc_ids:
            return
            
        try:
            self.collection.delete(ids=doc_ids)
            self.logger.info(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            raise
    
    def update_documents(
        self, 
        doc_ids: List[str], 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None
    ):
        """
        Update documents with specified IDs

        Args:
            doc_ids: List of document IDs
            texts: List of new texts
            metadatas: List of new metadata
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        
        try:
            self.collection.delete(ids=doc_ids)
            
            if metadatas is None:
                metadatas = [{}] * len(doc_ids)
            
            if self.use_custom_embeddings:
                embeddings = self._encode_text(texts)
                batch_embeddings = embeddings.tolist() if embeddings is not None else None
                
                if batch_embeddings is not None:
                    self.collection.add(
                        ids=doc_ids,
                        embeddings=batch_embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                else:
                    self.collection.add(
                        ids=doc_ids,
                        documents=texts,
                        metadatas=metadatas
                    )
            else:
                self.collection.add(
                    ids=doc_ids,
                    documents=texts,
                    metadatas=metadatas
                )
            
            self.logger.info(f"Updated {len(doc_ids)} documents")
        except Exception as e:
            self.logger.error(f"Failed to update documents: {e}")
            raise
    
    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            collection_metadata = {"hnsw:space": self.distance_function}
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.collection._embedding_function,
                metadata=collection_metadata
            )
            self.logger.info("Collection reset successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            raise