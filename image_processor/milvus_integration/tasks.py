from pymilvus import connections, utility, DataType, FieldSchema, Collection, CollectionSchema, IndexType, MilvusException
from django.conf import settings
import os

def addVectorsToMilvus( fileId,vectorEmbedding,eventId):
    collection = getCollection()
    if collection is None:
        print("Error: Collection is None")
        return
    data = [
        [fileId],  # fileId
        [vectorEmbedding],  # vectorEmbedding
        [eventId]  # eventId
    ]
    try:
        collection.insert(data)
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error while inserting data: {e}")


def create_index_if_not_exists(collection):
    try:
        # Check if the index already exists
        if not collection.has_index():
            print("Creating index for the collection...")
            # Define the index parameters
            index_params = {
                "index_type": "IVF_FLAT",  # Example index type
                "metric_type": "L2",       # Example metric type (Euclidean distance)
                "params": {"nlist": 128}   # Example parameters for IVF_FLAT
            }
            # Create the index on the `vectorEmbedding` field
            collection.create_index(field_name="vectorEmbedding", index_params=index_params)
            print("Index created successfully.")
        else:
            print("Index already exists.")
    except MilvusException as e:
        print(f"Error creating index: {e}")

def getCollection():
    print("Getting collection...")
    # Connect to Milvus
    milvus_config = settings.DATABASES['milvus']
    connections.connect(alias="default", host=milvus_config['HOST'], port=milvus_config['PORT'])
    
    if utility.has_collection('image_collection'):
        collection = Collection(name="image_collection")
        print(f"Collection 'image_collection' already exists.")
    else:
        fields = [
            FieldSchema(name="vectorId", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="fileId", dtype=DataType.INT64),
            FieldSchema(name="vectorEmbedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # Adjust dim as needed
            FieldSchema(name="eventId", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Collection for storing vector embeddings")
        collection_name = "image_collection"
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection 'image_collection' created.")
    
    # Create an index if it doesn't exist
    create_index_if_not_exists(collection)
    
    return collection

def get_all_data(request):
    collection = getCollection()
    collection.load()  # Load the collection into memory
    
    # Query all data from the collection
    results = collection.query(expr="vectorId>0", output_fields=["vectorId", "fileId", "vectorEmbedding", "eventId"]) # Adjust the query expression as needed
    print(results)
    # Prepare the data for the template
    results_list = []
    for result in results:
        results_list.append({
            'vectorId': result.get('vectorId', 'N/A'),  # Use .get() to avoid KeyError
            'fileId': result.get('fileId', 'N/A'),
            'vectorEmbedding': result.get('vectorEmbedding', 'N/A'),
            'eventId': result.get('eventId', 'N/A'),
        })
    
    # Render the template with the results
    return results_list

def delete_collection():
    try:
        collection = Collection(name="image_collection")
        collection.drop()
        print("Collection 'image_collection' deleted.")
    except MilvusException as e:
        print(f"Error deleting collection: {e}")