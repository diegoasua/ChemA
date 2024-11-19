import weaviate
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLQueryUtil:
    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        """
        Initialize the ChEMBL query utility.
        
        Args:
            weaviate_url (str): URL of the Weaviate instance
        """
        self.client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=weaviate_url,
                grpc_port=50051
            )
        )
        self.client.connect()
        self.collection = self.client.collections.get("ChEMBLCompound")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def semantic_search(self, 
                    query: str, 
                    limit: int = 5, 
                    additional_filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode(query)
            
            # Prepare and execute the query
            query_obj = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=limit,
                # Updated metadata configuration
                return_metadata=["distance"]  # Changed from weaviate.metadata.Metadata(distance=True)
            )
            
            if additional_filters:
                query_obj = query_obj.with_where(additional_filters)
                
            results = query_obj.objects
            
            # Convert to more friendly format
            return [
                {
                    **obj.properties,
                    "certainty": 1 - obj.metadata.distance
                }
                for obj in results
            ]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise

    def filter_by_phase(self, min_phase: int = 0, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Filter compounds by their clinical trial phase.
        
        Args:
            min_phase (int): Minimum clinical trial phase
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of matching compounds
        """
        try:
            where_filter = {
                "path": ["max_phase"],
                "operator": "GreaterThanEqual",
                "valueNumber": min_phase
            }
            
            results = (
                self.collection.query.get()
                .with_where(where_filter)
                .with_limit(limit)
                .objects
            )
            
            return [obj.properties for obj in results]
            
        except Exception as e:
            logger.error(f"Error in phase filter: {str(e)}")
            raise

    def get_similar_compounds(self, 
                            chembl_id: str, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find compounds similar to a given ChEMBL ID.
        
        Args:
            chembl_id (str): ChEMBL ID of the reference compound
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of similar compounds
        """
        try:
            # First, get the reference compound
            where_filter = {
                "path": ["molecule_chembl_id"],
                "operator": "Equal",
                "valueString": chembl_id
            }
            
            reference = (
                self.collection.query.get()
                .with_where(where_filter)
                .with_additional(["vector"])
                .objects
            )
            
            if not reference:
                raise ValueError(f"No compound found with ChEMBL ID: {chembl_id}")
            
            # Get the vector of the reference compound
            ref_vector = reference[0].metadata.vector
            
            # Find similar compounds
            similar = (
                self.collection.query.near_vector(
                    near_vector=ref_vector,
                    limit=limit + 1,
                    return_metadata=["distance"]
                )
                .objects
            )
            
            # Filter out the reference compound and convert to friendly format
            results = [
                {
                    **obj.properties,
                    "certainty": 1 - obj.metadata.distance
                }
                for obj in similar
                if obj.properties["molecule_chembl_id"] != chembl_id
            ][:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar compounds: {str(e)}")
            raise

def main():
    """Example usage of the query utility."""
    query_util = ChEMBLQueryUtil()
    
    # Example 1: Semantic search
    print("\nSemantic search for 'cancer treatment compounds':")
    results = query_util.semantic_search("cancer treatment compounds", limit=3)
    for result in results:
        print(f"- {result['pref_name']} ({result['molecule_chembl_id']})")
    
    # Example 2: Filter by phase
    print("\nCompounds in Phase 3 or higher:")
    results = query_util.filter_by_phase(min_phase=3, limit=3)
    for result in results:
        print(f"- {result['pref_name']} (Phase {result['max_phase']})")
    
    # Example 3: Similar compounds
    if results:  # Use the first compound from previous query as reference
        chembl_id = results[0]['molecule_chembl_id']
        print(f"\nCompounds similar to {results[0]['pref_name']}:")
        similar = query_util.get_similar_compounds(chembl_id, limit=3)
        for result in similar:
            print(f"- {result['pref_name']} ({result['molecule_chembl_id']})")

if __name__ == "__main__":
    main()