from fastapi import APIRouter
from schemas.schemas import populate_vectors_input, Health
from vector_database_ingestion_pipeline import VectorIngestion
from data_ingestion_pipeline import DataIngestionPipeline
from fastapi.responses import JSONResponse
from schemas.config import settings
import os

api_router = APIRouter()


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
    postgresDB_hostname = os.environ["postgresDB_hostname"]
    print(f"postgresDB_hostname: {postgresDB_hostname}",flush=True) 
    data_ingestion_pipeline = DataIngestionPipeline()
    open_source_mode = request.open_source_mode
    inject_vectors = VectorIngestion(open_source_mode)
        #Fetching all the records of python_stackOverFlow as a dataframe from the sqlite database
    sql_query = "select * from stackoverflow_posts"
    try:
        df = data_ingestion_pipeline.query_data(sql_query)
        print(f"Data fetched from the database\n Total count of data: {df.shape[0]}",flush=True)
    except Exception as e:
        print(f"Error in fetching data from the database: {str(e)}")
        return JSONResponse(status_code=201, content={"message": "Error in fetching data from the database"})

    #pass the dataframe to data parsing function, which parse the data and generate embeddings store it in the Vector database.
    batch_size = 256
    documents,metadatas,ids = inject_vectors.data_parsing(df)
    inject_vectors.data_insertion_to_vectordb(documents,metadatas,ids,batch_size)
   
    return JSONResponse(status_code=200, content={"message": "Vectors populated successfully"})

