
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from huggingface_hub import InferenceClient
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import voyageai
import uuid
import json
from tqdm import tqdm
import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

load_dotenv() 

class VectorIngestion:
  def __init__(self,open_source_mode):

    self.VOYAGEAI_API_KEY = os.getenv("VoyageAI_API_KEY")
    self.hf_client_API_KEY = os.getenv("HuggingFace_API_KEY")
    self.open_source_mode = open_source_mode
    self.hf_client = InferenceClient(api_key=self.hf_client_API_KEY)
    self.index_name = "chroma_vectorDB"


    if self.open_source_mode:
      print(f"Open Source Embedding Model Intialized...\n")
      embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
      collection_name = "hf_collection"

    else:
      # Initialize Voyageai
      print(f"VoyageAI Embedding Model Intialized...\n")
      print(f"VoyageAI API Key: {self.VOYAGEAI_API_KEY}")
      embeddings_model = VoyageAIEmbeddings(api_key = self.VOYAGEAI_API_KEY ,  model="voyage-large-2-instruct")
      collection_name = "voyageai_collection"

    # #initialize chromaDB
    self.vector_store = Chroma(
      collection_name=collection_name,
      embedding_function=embeddings_model,
      persist_directory="./data/chroma_db",  # Where to save data locally, remove if not necessary
    )

    print(f"ChromaDB Vector Store Intialized...")
    print(f"Index named {self.index_name} created...")
    print(f"Embedding Model Intialized...")
    print(f"Collection '{collection_name}' Intialized...\n")


  def data_parsing(self,df):
    print(f"creating the documents, metadata and uniques ids to insert into vector database")
    docs,metadatas,ids = [],[],[]
    for id, row in df.iterrows():

      docs.append(row['title'] + '\n\n' + row["question_body"])
      metadata = {"question_id":row["question_id"],
                  "tags":row['tags'],
                  "title":row["title"],
                  "question_body":row["question_body"],
                  "answer_ids" : json.dumps(row["answer_ids"])
                  }
      metadatas.append(metadata)
      ids.append(str(uuid.uuid4()))
    print(f"length of uniques question: {len(docs),len(metadatas),len(ids)}")
    return docs,metadatas,ids


  def generate_embeddings(self,documents,input_type):
      print(f"\nGenerating the embeddings of the {len(documents)} documents...")
      voyageai.api_key = self.VOYAGEAI_API_KEY
      vo = voyageai.Client()
      # Generate embeddings
      batch_size = 128
      embeddings = []

      for i in tqdm(range(0, len(documents), batch_size)):
          embeddings += vo.embed(
              documents[i:i + batch_size], model="voyage-large-2-instruct", input_type=input_type
          ).embeddings

      return embeddings


  # def data_insertion_to_vectordb(
  #           self,
  #           documents: List[str],
  #           metadatas: List[Dict[str, Any]],
  #           ids: List[str],
  #           batch_size: int = None
  #       ) -> bool:
  #           """
  #           Insert documents into vector database in batches.

  #           Args:
  #               documents: List of text documents
  #               metadatas: List of metadata dictionaries
  #               ids: List of document IDs
  #               batch_size: Optional batch size override

  #           Returns:
  #               bool: True if insertion was successful
  #           """
  #           if batch_size is None:
  #               batch_size = self.batch_size

  #           total_docs = len(documents)
  #           successful_insertions = 0

  #           # Process in batches with progress bar
  #           with tqdm(total=total_docs, desc="Inserting documents") as pbar:
  #               for i in range(0, total_docs, batch_size):
  #                   # Get batch slices
  #                   batch_docs = documents[i:i + batch_size]
  #                   batch_metadata = metadatas[i:i + batch_size]
  #                   batch_ids = ids[i:i + batch_size]

  #                   # Create Document objects for the batch
  #                   batch_documents = [
  #                       Document(
  #                           page_content=doc,
  #                           metadata=metadata,
  #                           id=doc_id
  #                       )
  #                       for doc, metadata, doc_id in zip(batch_docs, batch_metadata, batch_ids)
  #                   ]

  #                   try:
  #                       # Insert batch into vector store
  #                       resp = self.vector_store.add_documents(
  #                           documents=batch_documents,
  #                           ids=batch_ids
  #                       )
  #                       successful_insertions += len(resp)
  #                       pbar.update(len(batch_docs))

  #                   except Exception as e:
  #                       print(f"Error inserting batch {i//batch_size}: {str(e)}")
  #                       continue

  #           print(f"Successfully inserted {successful_insertions}/{total_docs} documents")



  # def process_batch(self,args: Tuple[List[str], List[Dict], List[str]]) -> int:
  #     """Process a single batch of documents."""
  #     batch_docs, batch_metadata, batch_ids = args
      
  #     # Create Document objects
  #     batch_documents = [
  #         Document(page_content=doc, metadata=metadata, id=doc_id)
  #         for doc, metadata, doc_id in zip(batch_docs, batch_metadata, batch_ids)
  #     ]
      
  #     try:
  #         resp = self.vector_store.add_documents(documents=batch_documents, ids=batch_ids)
  #         return len(resp)
  #     except Exception as e:
  #         print(f"Batch insertion error: {str(e)}")
  #         return 0

  # def data_insertion_to_vectordb(
  #         self,
  #         documents: List[str],
  #         metadatas: List[Dict[str, Any]],
  #         ids: List[str],
  #         batch_size: int = None,
  #         num_processes: int = None
  #     ) -> bool:
  #     """Insert documents into vector database using parallel processing."""
      
  #     batch_size = batch_size or self.batch_size
  #     num_processes = num_processes or max(1, cpu_count() - 1)
  #     total_docs = len(documents)
      
  #     # Prepare batches
  #     batches = [
  #         (
  #             documents[i:i + batch_size],
  #             metadatas[i:i + batch_size],
  #             ids[i:i + batch_size]
  #         )
  #         for i in range(0, total_docs, batch_size)
  #     ]
      
  #     # Process batches in parallel
  #     with Pool(num_processes) as pool:
  #         successful_insertions = sum(tqdm(
  #             pool.imap(self.process_batch, batches),
  #             total=len(batches),
  #             desc="Inserting documents"
  #         ))
      
  #     print(f"Successfully inserted {successful_insertions}/{total_docs} documents")
  #     return successful_insertions == total_docs


  def data_insertion_to_vectordb(
          self,
          documents: List[str],
          metadatas: List[Dict[str, Any]],
          ids: List[str],
          batch_size: int = None,
          max_workers: int = 4
      ) -> bool:
      """Insert documents into vector database using parallel processing."""
      
      batch_size = batch_size or self.batch_size
      total_docs = len(documents)
      successful_insertions = 0
      
      def process_batch(batch_idx: int) -> int:
          start_idx = batch_idx * batch_size
          end_idx = min(start_idx + batch_size, total_docs)
          
          batch_docs = documents[start_idx:end_idx]
          batch_metadata = metadatas[start_idx:end_idx]
          batch_ids = ids[start_idx:end_idx]
          
          batch_documents = [
              Document(page_content=doc, metadata=metadata, id=doc_id)
              for doc, metadata, doc_id in zip(batch_docs, batch_metadata, batch_ids)
          ]
          
          try:
              resp = self.vector_store.add_documents(documents=batch_documents, ids=batch_ids)
              return len(resp)
          except Exception as e:
              print(f"Batch insertion error: {str(e)}")
              return 0
      
      num_batches = (total_docs + batch_size - 1) // batch_size
      
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
          results = list(tqdm(
              executor.map(process_batch, range(num_batches)),
              total=num_batches,
              desc="Inserting documents"
          ))
      
      successful_insertions = sum(results)
      print(f"Successfully inserted {successful_insertions}/{total_docs} documents")
      return successful_insertions == total_docs

# if __name__ == "__main__":
#     from data_ingestion_pipeline import DataIngestionPipeline
#     data_ingestion_pipeline = DataIngestionPipeline()
#     open_source_mode = True
#     inject_vectors = VectorIngestion(open_source_mode)
#     #Fetching all the records of python_stackOverFlow as a dataframe from the sqlite database
#     sql_query = "select * from stackoverflow_posts"
#     df = data_ingestion_pipeline.query_data(sql_query)
#     print(f"Data fetched from the sqlite database\n",flush=True)
#     print(f"Dataframe shape: {df.shape}\n",flush=True)

#     #pass the dataframe to data parsing function, which parse the data and generate embeddings store it in the Vector database.
#     batch_size = 256
#     documents,metadatas,ids = inject_vectors.data_parsing(df)
#     inject_vectors.data_insertion_to_vectordb(documents,metadatas,ids,batch_size,3)