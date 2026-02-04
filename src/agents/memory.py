from typing import Any, List, Tuple

from tools.dense_retriever import DenseRetriever

class SimpleMemory:
    def __init__(self, memories: str = None):
        self.memories = [memories] if memories else []
        
    def add(self, memory: str):
        self.memories.append(memory)
    
    def get(self) -> str:
        return '\n'.join(self.memories)

    def kpop(self, k: int = 1):
        if k <= len(self.memories):
            self.memories = self.memories[:-k]
        else:
            self.memories = []

class KVMemory:
    def __init__(self, retriever_config, memories: List[dict] = None, key_field: str = "query", value_field: str = "response", use_text_strategy: bool = True):
        """
        Initialization of KVMemory Class

        :param retriever_config: config of retriever
        :param memories: initial memories
        """
        self.retriever = DenseRetriever(**retriever_config)
        self.retriever.reset_collection()
        self.key2id_value = {}
        self._counter = 0
        self.key_field = key_field
        self.value_field = value_field
        self.use_text_strategy = use_text_strategy
        if memories:
            texts = []
            for memory in memories:
                _id = self.new_id()
                key = memory.get(key_field)
                value = memory.get(value_field)
                if key and value:
                    self.key2id_value[key] = [_id, value]
                memory['id'] = _id
                memory['text'] = self._generate_semantic_text(key, value, use_text_strategy=self.use_text_strategy)
                texts.append(memory)
            self.retriever.build_index(data=texts)
    
    def new_id(self):
        self._counter += 1
        return str(self._counter)

    def _generate_semantic_text(self, key: str, value: Any, use_text_strategy: bool = True) -> str:
        """
        Generate semantic text for a given key and value.
        """
        if use_text_strategy:
            return f"Query: {key}\nResponse: {value}"
        else:
            return value
    
    def get(self, key: str) -> Any:
        """
        return the value for a given key

        :param key: key
        :return: value, or None if not found
        """
        id_value = self.key2id_value.get(key, None)
        if id_value:
            return id_value[1]
        else:
            return None

    def search(self, query, top_k: int = 3) -> List[Tuple]:
        results = self.retriever.search(query, top_k=top_k, return_scores=True)
        results = [(x['data'], x['score']) for x in results]
        return results

    def add_key_value(self, key: str, value: Any):
        """
        add or update a key-value pair

        :param key: key
        :param value: value
        """
        _id = self.new_id()
        self.key2id_value[key] = [_id, value]
        memory = {
            self.key_field: key,
            self.value_field: value,
            'text': self._generate_semantic_text(key, value, use_text_strategy=self.use_text_strategy),
            'id': str(_id)
        }
        self.retriever.add_documents([memory])
    
    def add_item(self, item: dict):
        """
        add a new item containing key and value

        :param item: item dict
        """
        key = item.get(self.key_field)
        value = item.get(self.value_field)
        if key and value:
            _id = self.new_id()
            self.key2id_value[key] = [_id, value]
            item['text'] = self._generate_semantic_text(key, value, use_text_strategy=self.use_text_strategy)
            item['id'] = _id
            self.retriever.add_documents([item])


class NullMemory:
    """Fallback memory that performs no storage or retrieval."""

    def search(self, query, top_k: int = 3) -> List[Tuple]:
        return []

    def add_key_value(self, key: str, value: Any):
        return None
