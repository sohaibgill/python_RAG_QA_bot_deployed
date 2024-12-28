from fastapi import APIRouter
from src.schemas.schemas import populate_vectors_input, Health, queryInput
from src.vector_database_ingestion_pipeline import VectorIngestion
from src.data_ingestion_pipeline import DataIngestionPipeline
from fastapi.responses import JSONResponse
from src.schemas.config import settings
from src.inference_pipeline import InferenceClass
import os

open_source_mode=False
api_router = APIRouter()
data_ingestion_pipeline = DataIngestionPipeline()
inference_instance = InferenceClass(
ingestion_pipeline=data_ingestion_pipeline,
open_source_mode=open_source_mode,
log_file='inference_logs.csv')
inject_vectors = VectorIngestion(open_source_mode)


@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = Health(
        name=settings.PROJECT_NAME, api_version="0.1"
    )

    return health.dict()

@api_router.post("/populate_vectors")
def populate_vectors(request: populate_vectors_input):
    """
    Populate vectors in the vector database.
    """
    postgresDB_hostname = os.environ["hostname"]
    print(f"postgresDB_hostname: {postgresDB_hostname}",flush=True) 
    open_source_mode = request.open_source_mode
    
        #Fetching all the records of python_stackOverFlow as a dataframe from the sqlite database
    sql_query = "select * from stackoverflow_posts limit 1000"
    try:
        df = data_ingestion_pipeline.query_data(sql_query)
        print(f"Data fetched from the database\n Total count of data: {df.shape[0]}",flush=True)
    except Exception as e:
        print(f"Error in fetching data from the database: {str(e)}")
        return JSONResponse(status_code=201, content={"message": "Error in fetching data from the database"})

    #pass the dataframe to data parsing function, which parse the data and generate embeddings store it in the Vector database.
    batch_size = 256
    workers = os.environ["workers_count"]
    documents,metadatas,ids = inject_vectors.data_parsing(df)
    inject_vectors.data_insertion_to_vectordb(documents,metadatas,ids,batch_size,int(workers))
   
    return JSONResponse(status_code=200, content={"message": "Vectors populated successfully"})

@api_router.post("/query")
def query(request: queryInput):
    """
    Query the vector database.
    """
    query = request.query
    verbose = request.verbose
    stream = request.stream


    try:
        response = inference_instance.llm_call(query,verbose,stream)
    
    except Exception as e:
        print(f"Error in generating response for the query: {str(e)}")
        return JSONResponse(status_code=201, content={"message": "Error in generating response for the query"})

    return JSONResponse(status_code=200, content={"message": response})