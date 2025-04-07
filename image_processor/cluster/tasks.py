from pymilvus import connections, Collection
import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from chinese_whispers import chinese_whispers, aggregate_clusters
from milvus_integration import tasks

def cluster_embeddings_chinese_whispers(specific_embedding, event_id, similarity_threshold=0.95):
    # Load the collection
    collection = tasks.getCollection()
    collection.load()

    # Retrieve all stored embeddings and metadata
    results = collection.query(
        expr=f"eventId == {event_id}",
        output_fields=["vectorEmbedding", "fileId", "eventId"],
        limit=10000
    )
    if not results:
        raise ValueError("No embeddings found in Milvus collection.")

    # Extract embeddings, file IDs, and event IDs
    embeddings = [item["vectorEmbedding"] for item in results]
    file_ids = [item["fileId"] for item in results]

    # Convert embeddings to NumPy array
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Create graph for Chinese Whispers clustering
    G = nx.Graph()
    for i in range(len(embeddings_np)):
        G.add_node(i, file_id=file_ids[i], embedding=embeddings_np[i])

    # Add edges based on cosine similarity
    for i in range(len(embeddings_np)):
        for j in range(i + 1, len(embeddings_np)):
            similarity = 1 - cosine(embeddings_np[i], embeddings_np[j])  # Cosine similarity
            if similarity >= similarity_threshold:
                G.add_edge(i, j, weight=similarity)

    # Run Chinese Whispers algorithm
    chinese_whispers(G, iterations=20)

    # Map nodes to clusters
    clusters = {}
    for node, data in G.nodes(data=True):
        label = G.nodes[node]['label']  # Cluster label assigned by Chinese Whispers
        clusters.setdefault(label, []).append(node)

    # Find direct matches for the specific embedding
    direct_matches = []
    for i, embedding in enumerate(embeddings_np):
        similarity = 1 - cosine(specific_embedding, embedding)
        if similarity >= similarity_threshold:
            direct_matches.append(i)

    # If direct matches exist, find their cluster
    specific_cluster_file_ids = []
    if direct_matches:
        for match_index in direct_matches:
            cluster_label = G.nodes[match_index]['label']
            cluster_nodes = clusters[cluster_label]
            specific_cluster_file_ids.extend(
                G.nodes[node]['file_id'] for node in cluster_nodes
            )
    else:
        raise ValueError("No strong matches found for the specific embedding.")

    # Remove duplicates and return
    specific_cluster_file_ids = list(set(specific_cluster_file_ids))
    return specific_cluster_file_ids
