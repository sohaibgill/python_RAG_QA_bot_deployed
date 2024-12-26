from pydantic import BaseModel

class populate_vectors_input(BaseModel):
    open_source_mode: str

class query_input(BaseModel):
    query: str
    verbose: bool
    stream: bool
    open_source_mode: bool

class Health(BaseModel):
    name: str
    api_version: str

class queryInput(BaseModel):
    query: str
    verbose: bool
    stream: bool