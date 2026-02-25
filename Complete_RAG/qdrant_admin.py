import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

class QdrantAdmin:
    """ Class to manage the connection with Qdrant and the operations on the database.
    Since there are parameters that are specific to the format of the metadata generated 
    by LangChain loaders, we ensure a uniform schema across different file types 
    (PDF, TXT, DOCX) through a normalization process. """

    def __init__(self, url: str, collection_name: str, vector_size: int):
        self.client = QdrantClient(url = url) 
        self.collection_name = collection_name
        self.vector_size = vector_size
    
    def exists_collection(self, collection_name = None) -> bool:
        if not collection_name:
            collection_name = self.collection_name
        return self.client.collection_exists(collection_name)
    
    def delete_collection(self, collection_name = None):
        if not collection_name:
            collection_name = self.collection_name
        if self.exists_collection(collection_name):
            self.client.delete_collection(collection_name)

    def create_collection(self, collection_name = None, vector_size = None):
        """ Function to create a specific collection if it does not already exist """
        if not collection_name:
            collection_name = self.collection_name
        if not vector_size:
            vector_size = self.vector_size

        if not self.exists_collection(collection_name):
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = vector_size, 
                                              distance = Distance.COSINE)
            )
    
    def chunks_normalization(self, chunks):
        """ This function is needed to create uniform metadatas for all the types of documents """
        for chunk in chunks:
            original_source = chunk.metadata.get("source", "unknown")
            chunk.metadata["clean_filename"] = os.path.basename(original_source) # from ./...txt to ...txt
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = "unknown"
                
            if original_source.endswith(".pdf"):
                chunk.metadata["file_type"] = "pdf"
            elif original_source.endswith(".txt"):
                chunk.metadata["file_type"] = "txt"
            elif original_source.endswith(".docx"):
                chunk.metadata["file_type"] = "docx"
        
        return chunks

    def num_total_points(self, collection_name = None):
        if not collection_name:
            collection_name = self.collection_name
        return self.client.count(collection_name).count
    
    def list_of_points_with_payload(self, collection_name = None):
        # https://python-client.qdrant.tech/qdrant_client.qdrant_client
        if not collection_name:
            collection_name = self.collection_name
        records, _= self.client.scroll(
            collection_name = collection_name,
            limit = 1000,
            with_payload = True,
            with_vectors = False)
        
        return records
    
    def unique_filenames(self, collection_name = None) -> list:
        if not collection_name:
            collection_name = self.collection_name
        records = self.list_of_points_with_payload(collection_name)
        filenames = [
            p.payload["metadata"]["clean_filename"] 
            for p in records 
            if p.payload and "metadata" in p.payload and "clean_filename" in p.payload["metadata"]
        ]
        unique_filenames = list(set(filenames))

        return unique_filenames
    
    def is_file_in_db(self, filename: str, collection_name = None) -> bool:
        if not collection_name:
            collection_name = self.collection_name
        return filename in self.unique_filenames(collection_name)

    def collection_info(self, collection_name = None):
        if not collection_name:
            collection_name = self.collection_name
        print(f"Total number of points:\n\n{self.num_total_points(collection_name)}\n\n")
        unique_names = self.unique_filenames(collection_name)
        print(f"There are {len(unique_names)} documents ingested, with titles:\n\n")
        for unique_name in unique_names:
            print(unique_name)
        print(f"General info: \n\n{self.client.get_collection(collection_name)}\n\n")

    def remove_a_file(self, filename: str, collection_name = None):
        if not collection_name:
            collection_name = self.collection_name
        if self.is_file_in_db(filename, collection_name):
            self.client.delete(
                collection_name = collection_name,
                points_selector = Filter(
                    must = [
                        FieldCondition(
                            key = "metadata.clean_filename", 
                            match = MatchValue(value = filename),
                        ),
                    ],
                )
            )
  