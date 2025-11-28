import os
import logging
from pinecone import Pinecone
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self):
        # Initialize OpenAI client for embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize Pinecone with new SDK
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY environment variable is required. "
                "Get your API key from https://app.pinecone.io/"
            )
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get or create index
        index_name = os.getenv("PINECONE_INDEX_NAME", "london-law-firms")
        dimension = 1536  # text-embedding-3-small dimension
        
        # Check if index exists (using new API)
        # Note: Index creation should be done via CLI, but we check here
        if not self.pc.has_index(index_name):
            raise RuntimeError(
                f"Index '{index_name}' does not exist. "
                f"Please create it using the Pinecone CLI:\n"
                f"  pc index create --name {index_name} --dimension {dimension} "
                f"--metric cosine --cloud aws --region us-east-1"
            )
        
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        # Use a default namespace for this application
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")

    def _get_embedding(self, text):
        """Generate embedding using OpenAI API."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def add_firm(self, firm_data):
        vector_text = f"""
        Firm: {firm_data['firm_name']}
        Tone: {firm_data['firm_tone']}
        Keywords: {', '.join(firm_data['hiring_keywords'])}
        Lifestyle: {firm_data['lifestyle_summary']}
        Sectors: {', '.join(firm_data['sector_focus'])}
        """

        # Generate embedding
        embedding = self._get_embedding(vector_text)
        
        # Prepare metadata (Pinecone requires flat structure, no nested objects)
        metadata = {
            "name": firm_data['firm_name'],
            "tone": firm_data['firm_tone'],
            "keywords": ", ".join(firm_data['hiring_keywords']),
            "wins": ", ".join(firm_data['recent_wins']),
            "motto": firm_data['motto'],
            "document": vector_text  # Store the full text in metadata for retrieval
        }
        
        # Generate ID
        firm_id = firm_data['firm_name'].replace(" ", "_").lower()
        
        # Upsert to Pinecone using new API (vector-based)
        # Note: New SDK uses different structure - using vectors parameter
        self.index.upsert(
            vectors=[{
                "id": firm_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=self.namespace
        )
        print(f"Saved {firm_data['firm_name']} to Pinecone Database.", flush=True)
    
    def add_jobs(self, firm_name: str, jobs_data: list):
        """
        Store job listings as a metric [keyword/tone] in the vector database.
        
        Args:
            firm_name: Name of the law firm
            jobs_data: List of job dictionaries with title, company, location, summary, link, platform
        """
        if not jobs_data:
            logger.info(f"No jobs to store for {firm_name}")
            return
        
        # Create a comprehensive text representation of all jobs
        jobs_text = f"Current Job Openings at {firm_name}:\n\n"
        for job in jobs_data:
            jobs_text += f"Title: {job.get('title', 'N/A')}\n"
            jobs_text += f"Location: {job.get('location', 'N/A')}\n"
            jobs_text += f"Summary: {job.get('summary', 'N/A')}\n"
            jobs_text += f"Platform: {job.get('platform', 'N/A')}\n"
            jobs_text += f"Link: {job.get('link', 'N/A')}\n"
            jobs_text += "---\n\n"
        
        # Generate embedding for the jobs data
        embedding = self._get_embedding(jobs_text)
        
        # Prepare metadata
        # Store job titles, keywords extracted from job descriptions, and platforms
        job_titles = [job.get('title', '') for job in jobs_data]
        platforms = list(set([job.get('platform', '') for job in jobs_data]))
        
        metadata = {
            "name": firm_name,
            "type": "jobs",  # Mark this as job data
            "job_count": str(len(jobs_data)),
            "job_titles": " | ".join(job_titles[:20]),  # Limit to first 20 titles
            "platforms": ", ".join(platforms),
            "document": jobs_text,
            "tone": "hiring",  # Indicate this is hiring-related content
            "keywords": "jobs, hiring, recruitment, positions, vacancies"  # Standard job-related keywords
        }
        
        # Generate ID for job data (separate from firm data)
        jobs_id = f"{firm_name.replace(' ', '_').lower()}_jobs"
        
        # Upsert to Pinecone
        self.index.upsert(
            vectors=[{
                "id": jobs_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=self.namespace
        )
        print(f"Saved {len(jobs_data)} jobs for {firm_name} to Pinecone Database.", flush=True)

    def add_career_insights(self, firm_name: str, career_data: dict):
        """
        Store structured insights cultured from the firm's own career pages.
        """
        if not career_data:
            logger.info(f"No career page insights to store for {firm_name}")
            return

        openings = career_data.get("current_openings") or []
        opening_blocks = []
        for opening in openings:
            parts = [
                f"Title: {opening.get('title', 'N/A')}",
                f"Practice Area: {opening.get('practice_area', '')}",
                f"Location: {opening.get('location', '')}",
                f"Experience: {opening.get('experience_level', '')}",
                f"Deadline: {opening.get('application_deadline', '')}",
                f"Apply: {opening.get('application_link', '')}",
                f"Notes: {opening.get('notes', '')}",
            ]
            opening_blocks.append("\n".join([p for p in parts if p and not p.endswith(': ')]))

        insights_text = f"""
        Career Page Snapshot for {firm_name}
        Hiring focus: {career_data.get('hiring_focus', '')}
        Application routes: {', '.join(career_data.get('application_channels', []))}
        Benefits highlighted: {', '.join(career_data.get('benefits', []))}
        Interview process: {career_data.get('interview_process', '')}
        Candidate tips: {', '.join(career_data.get('candidate_tips', []))}

        Openings:
        {chr(10).join(opening_blocks)}
        """

        embedding = self._get_embedding(insights_text)

        metadata = {
            "name": firm_name,
            "type": "career_page",
            "hiring_focus": career_data.get("hiring_focus", ""),
            "application_channels": " | ".join(career_data.get("application_channels", [])),
            "benefits": " | ".join(career_data.get("benefits", [])),
            "interview_process": career_data.get("interview_process", ""),
            "candidate_tips": " | ".join(career_data.get("candidate_tips", [])),
            "openings_count": str(len(openings)),
            "document": insights_text,
            "keywords": "career, openings, application guidance"
        }

        vector_id = f"{firm_name.replace(' ', '_').lower()}_career"

        self.index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=self.namespace
        )
        print(f"Saved career page insights for {firm_name} to Pinecone Database.", flush=True)

    @property
    def collection(self):
        """Compatibility layer to maintain existing interface."""
        return PineconeCollectionAdapter(self.index, self._get_embedding, self.namespace)


class PineconeCollectionAdapter:
    """Adapter class to make Pinecone work like ChromaDB collection interface."""
    
    def __init__(self, index, embedding_fn, namespace):
        self.index = index
        self._get_embedding = embedding_fn
        self.namespace = namespace
    
    def query(self, query_texts, n_results=10, include=None):
        """
        Query Pinecone index using query() method (for vector-based queries).
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return
            include: List of what to include (e.g., ['documents', 'metadatas', 'distances'])
        
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', 'ids' lists
        """
        if not query_texts:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        # Generate embedding for the query
        query_embedding = self._get_embedding(query_texts[0])
        
        # Query Pinecone using new API (vector-based query)
        # Note: For custom embeddings, we use query() with vector parameter
        # For integrated embeddings, you would use search() with text inputs
        results = self.index.query(
            namespace=self.namespace,
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        # Format response to match ChromaDB format
        documents = []
        metadatas = []
        distances = []
        ids = []
        
        if results.get("matches"):
            for match in results["matches"]:
                ids.append(match["id"])
                metadata = match.get("metadata", {})
                metadatas.append(metadata)
                # Extract document from metadata if stored there
                documents.append(metadata.get("document", ""))
                # Pinecone returns similarity scores (higher is better), ChromaDB uses distances (lower is better)
                # Convert similarity to distance: distance = 1 - similarity
                similarity = match.get("score", 0.0)
                distances.append(1.0 - similarity)
        
        return {
            "documents": [documents],  # ChromaDB returns nested list
            "metadatas": [metadatas],
            "distances": [distances],
            "ids": [ids]
        }
    
    def get(self, include=None, ids=None):
        """
        Get all vectors or specific vectors by IDs.
        
        Args:
            include: List of what to include (e.g., ['documents', 'metadatas'])
            ids: Optional list of IDs to fetch
        
        Returns:
            Dictionary with 'documents', 'metadatas', 'ids' lists
        """
        documents = []
        metadatas = []
        result_ids = []
        
        if ids:
            # Fetch specific IDs using new API
            fetch_results = self.index.fetch(
                namespace=self.namespace,
                ids=ids
            )
            if fetch_results.get("vectors"):
                for vector_id, vector_data in fetch_results["vectors"].items():
                    result_ids.append(vector_id)
                    metadata = vector_data.get("metadata", {})
                    metadatas.append(metadata)
                    documents.append(metadata.get("document", ""))
        else:
            # Fetch all vectors using list() method (new API)
            # Use pagination to get all IDs
            all_ids = []
            pagination_token = None
            
            while True:
                list_result = self.index.list(
                    namespace=self.namespace,
                    limit=1000,
                    pagination_token=pagination_token
                )
                
                # Extract IDs from the result
                if hasattr(list_result, 'vectors'):
                    all_ids.extend([vec.id for vec in list_result.vectors])
                elif isinstance(list_result, dict) and 'vectors' in list_result:
                    all_ids.extend([vec.get('id', vec['id']) for vec in list_result['vectors']])
                else:
                    # Fallback: try to get IDs from the result object
                    try:
                        all_ids.extend([r.id for r in list_result])
                    except:
                        break
                
                # Check for pagination
                if hasattr(list_result, 'pagination') and list_result.pagination:
                    if hasattr(list_result.pagination, 'next') and list_result.pagination.next:
                        pagination_token = list_result.pagination.next
                    else:
                        break
                elif isinstance(list_result, dict) and 'pagination' in list_result:
                    if list_result['pagination'].get('next'):
                        pagination_token = list_result['pagination']['next']
                    else:
                        break
                else:
                    break
            
            # Fetch all records by their IDs
            if all_ids:
                # Fetch in batches (Pinecone has limits)
                batch_size = 100
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    fetch_results = self.index.fetch(
                        namespace=self.namespace,
                        ids=batch_ids
                    )
                    if fetch_results.get("vectors"):
                        for vector_id, vector_data in fetch_results["vectors"].items():
                            result_ids.append(vector_id)
                            metadata = vector_data.get("metadata", {})
                            metadatas.append(metadata)
                            documents.append(metadata.get("document", ""))
            else:
                # Fallback: if list() doesn't work, use query with dummy vector
                stats = self.index.describe_index_stats()
                total_vectors = stats.get("total_vector_count", 0)
                if isinstance(total_vectors, dict):
                    total_vectors = total_vectors.get(self.namespace, {}).get("vector_count", 0)
                
                if total_vectors > 0:
                    query_results = self.index.query(
                        namespace=self.namespace,
                        vector=[0.0] * 1536,  # Dummy zero vector
                        top_k=min(10000, total_vectors),
                        include_metadata=True
                    )
                    
                    if query_results.get("matches"):
                        for match in query_results["matches"]:
                            result_ids.append(match["id"])
                            metadata = match.get("metadata", {})
                            metadatas.append(metadata)
                            documents.append(metadata.get("document", ""))
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "ids": result_ids
        }
