from pymilvus import connections, Collection
import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from chinese_whispers import chinese_whispers, aggregate_clusters
from milvus_integration import tasks

def cluster_embeddings_chinese_whispers(specific_embedding,event_id):
    # Load the collection
    collection = tasks.getCollection()
    collection.load()  

    # Retrieve all stored embeddings and metadata
    results = collection.query(
        expr=f"eventId == {event_id}",
        output_fields=["vectorEmbedding", "fileId", "eventId"],
        limit=10000
    )
    print("1")
    if not results:
        raise ValueError("No embeddings found in Milvus collection.")
    print("2")
    # Extract embeddings, file IDs, and event IDs
    embeddings = [item["vectorEmbedding"] for item in results]
    file_ids = [item["fileId"] for item in results]
    print("3")
    # Convert embeddings to NumPy array
    embeddings_np = np.array(embeddings, dtype=np.float32)
    print("4")
    # Create graph for Chinese Whispers clustering
    G = nx.Graph()
    for i in range(len(embeddings_np)):
        G.add_node(i, file_id=file_ids[i], embedding=embeddings_np[i])
    print("5")
    # Add edges based on cosine similarity
    for i in range(len(embeddings_np)):
        for j in range(i + 1, len(embeddings_np)):
            similarity = 1 - cosine(embeddings_np[i], embeddings_np[j])  # Cosine similarity
            G.add_edge(i, j, weight=similarity)
    print("6")
    # Run Chinese Whispers algorithm
    chinese_whispers(G, iterations=20)
    clusters = aggregate_clusters(G)
    print("7")
    # Convert specific embedding to NumPy array
    specific_embedding = np.array(specific_embedding, dtype=np.float32)
    print("8")
    # Find the cluster that contains the specific embedding
    specific_embedding /= np.linalg.norm(specific_embedding)  # Normalize

    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)  # Normalize

    specific_cluster = None
    try:
        for i, embedding in enumerate(embeddings_np):
            if 1 - cosine(specific_embedding, embedding) > 0.99:  # High similarity
                specific_cluster = clusters[i]
                break
    except Exception as e:
        print(str())


    if specific_cluster is None:
        print("Specific embedding not found in any cluster.")
        return None

    print(f"Specific embedding belongs to cluster: {specific_cluster}")

    print("10")
    # Retrieve all file IDs in the same cluster
    clustered_file_ids = [file_ids[i] for i, cluster in clusters.items() if cluster == specific_cluster]
    print("11")
    return {"cluster_id": specific_cluster, "file_ids": clustered_file_ids}