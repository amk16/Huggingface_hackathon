import os
import chromadb
from chromadb.utils import embedding_functions


class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        self.collection = self.client.get_or_create_collection(
            name="london_law_firms",
            embedding_function=self.openai_ef
        )

    def add_firm(self, firm_data):
        vector_text = f"""
        Firm: {firm_data['firm_name']}
        Tone: {firm_data['firm_tone']}
        Keywords: {', '.join(firm_data['hiring_keywords'])}
        Lifestyle: {firm_data['lifestyle_summary']}
        Sectors: {', '.join(firm_data['sector_focus'])}
        """

        self.collection.add(
            documents=[vector_text],
            metadatas=[{
                "name": firm_data['firm_name'],
                "tone": firm_data['firm_tone'],
                "keywords": ", ".join(firm_data['hiring_keywords']),
                "wins": ", ".join(firm_data['recent_wins']),
                "motto": firm_data['motto']
            }],
            ids=[firm_data['firm_name'].replace(" ", "_").lower()]
        )
        print(f"Saved {firm_data['firm_name']} to RAG Database.")
