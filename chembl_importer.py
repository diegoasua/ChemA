import weaviate
from weaviate.util import generate_uuid5
from chembl_webresource_client.new_client import new_client
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLWeaviateImporter:
    def __init__(self, weaviate_url="http://localhost:8080", batch_size=100):
        """
        Initialize the ChEMBL to Weaviate importer.
        
        Args:
            weaviate_url (str): URL of the Weaviate instance
            batch_size (int): Number of records to process in each batch
        """
        self.client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=weaviate_url,
                grpc_port=50051
            )
        )
        self.client.connect()
        # Fix: Initialize new_client as a class instance
        self.chembl_client = new_client
        self.chembl_client.new_client_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.batch_size = batch_size
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_schema(self):
        """Create the Weaviate schema for ChEMBL compounds."""
        properties = [
            weaviate.Property(name="molecule_chembl_id", data_type=weaviate.DataType.TEXT),
            weaviate.Property(name="pref_name", data_type=weaviate.DataType.TEXT),
            weaviate.Property(name="molecule_type", data_type=weaviate.DataType.TEXT),
            weaviate.Property(name="max_phase", data_type=weaviate.DataType.NUMBER),
            weaviate.Property(name="therapeutic_flag", data_type=weaviate.DataType.BOOLEAN),
            weaviate.Property(name="structure_type", data_type=weaviate.DataType.TEXT),
            weaviate.Property(name="description", data_type=weaviate.DataType.TEXT),
        ]

        class_obj = weaviate.Collection(
            name="ChEMBLCompound",
            description="A chemical compound from ChEMBL database",
            properties=properties,
            vectorizer_config=weaviate.config.Configure.Vectorizer.none()
        )
        
        try:
            self.client.collections.create(class_obj)
            logger.info("Schema created successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            raise

    def fetch_chembl_data(self, limit=1000):
        """
        Fetch compound data from ChEMBL.
        
        Args:
            limit (int): Number of compounds to fetch
        
        Returns:
            list: List of compound dictionaries
        """
        try:
            molecule = self.chembl_client.molecule
            compounds = molecule.filter(max_phase__gte=0).order_by('-max_phase')[:limit]
            logger.info(f"Fetched {len(compounds)} compounds from ChEMBL")
            return compounds
        except Exception as e:
            logger.error(f"Error fetching ChEMBL data: {str(e)}")
            raise

    def create_compound_description(self, compound):
        """
        Create a textual description of the compound for embedding.
        
        Args:
            compound (dict): Compound data
        
        Returns:
            str: Textual description of the compound
        """
        description = f"This is a {compound.get('molecule_type', 'compound')} "
        if compound.get('pref_name'):
            description += f"named {compound['pref_name']} "
        description += f"with ChEMBL ID {compound['molecule_chembl_id']}. "
        
        if compound.get('max_phase'):
            phase_desc = {
                0: "not tested in clinical trials",
                1: "in Phase I clinical trials",
                2: "in Phase II clinical trials",
                3: "in Phase III clinical trials",
                4: "approved for clinical use"
            }
            description += phase_desc.get(compound['max_phase'], "")
        
        return description.strip()

    def process_batch(self, compounds, collection):
        """
        Process a batch of compounds and import them into Weaviate.
        
        Args:
            compounds (list): List of compound dictionaries to process
            collection: Weaviate collection object
        """
        with collection.batch.dynamic() as batch:
            for compound in compounds:
                description = self.create_compound_description(compound)
                
                # Generate embedding for the description
                embedding = self.model.encode(description)
                
                # Prepare data object
                data_object = {
                    "molecule_chembl_id": compound["molecule_chembl_id"],
                    "pref_name": compound.get("pref_name", ""),
                    "molecule_type": compound.get("molecule_type", ""),
                    "max_phase": compound.get("max_phase", 0),
                    "therapeutic_flag": compound.get("therapeutic_flag", False),
                    "structure_type": compound.get("structure_type", ""),
                    "description": description
                }
                
                # Add object to Weaviate
                try:
                    batch.add_object(
                        properties=data_object,
                        vector=embedding.tolist(),
                        uuid=generate_uuid5(compound["molecule_chembl_id"])
                    )
                except Exception as e:
                    logger.error(f"Error adding compound {compound['molecule_chembl_id']}: {str(e)}")

    def import_data(self, limit=1000):
        """
        Main method to import ChEMBL data into Weaviate.
        
        Args:
            limit (int): Number of compounds to import
        """
        try:
            # Create schema if it doesn't exist
            try:
                collection = self.client.collections.get("ChEMBLCompound")
            except:
                self.create_schema()
                collection = self.client.collections.get("ChEMBLCompound")
            
            # Fetch data from ChEMBL
            compounds = self.fetch_chembl_data(limit)
            
            # Process compounds in batches
            for i in tqdm(range(0, len(compounds), self.batch_size)):
                batch = compounds[i:i + self.batch_size]
                self.process_batch(batch, collection)
                time.sleep(0.1)  # Small delay to prevent overwhelming the API
            
            logger.info(f"Successfully imported {len(compounds)} compounds into Weaviate")
            
        except Exception as e:
            logger.error(f"Error during import: {str(e)}")
            raise

def main():
    """Main function to run the import process."""
    # Initialize importer
    importer = ChEMBLWeaviateImporter(
        weaviate_url="http://localhost:8080",
        batch_size=100
    )
    
    # Import data
    importer.import_data(limit=1000)  # Adjust limit as needed

if __name__ == "__main__":
    main()