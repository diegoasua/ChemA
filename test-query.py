from chembl_query import ChEMBLQueryUtil

# Initialize the query utility
query_util = ChEMBLQueryUtil()

results = query_util.semantic_search("compounds effective against breast cancer", limit=5)
for result in results:
    print(f"Name: {result['pref_name']}")
    print(f"ChEMBL ID: {result['molecule_chembl_id']}")
    print(f"Phase: {result['max_phase']}")
    print("---")
