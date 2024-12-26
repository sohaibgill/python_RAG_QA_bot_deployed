# Python RAG QA Bot Deployed

This project is a Retrieval-Augmented Generation (RAG) Question Answering (QA) bot deployed using FastAPI. The bot is designed to generate responses to user queries related to Python by leveraging a combination of retrieval and generation techniques.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Docker Deployment](#docker-deployment)
- [Project Components](#project-components)
- [Logs](#logs)
- [Database](#database)
- [Environment Variables](#environment-variables)
- [Ignored Files](#ignored-files)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Structure

## Setup Instructions

### Prerequisites

- Python 3.12.1
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:

    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:

    Create a [.env](http://_vscodecontentref_/7) file in the root directory with the following content:

    ```env
    HuggingFace_API_KEY= "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"
    VoyageAI_API_KEY="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"
    ```

### Running the Application

1. Start the FastAPI application:

    ```sh
    uvicorn app:app --host 0.0.0.0 --port 8001 --reload
    ```

2. Access the API documentation at `http://localhost:8001/docs`.

### Docker Deployment

1. Build the Docker image:

    ```sh
    docker build -t python_rag_qa_bot .
    ```

2. Run the Docker container:

    ```sh
    docker run -p 8001:8001 python_rag_qa_bot
    ```

## Project Components

### [app.py](http://_vscodecontentref_/8)

The main entry point for the FastAPI application. It sets up the FastAPI app, configures CORS middleware, and includes the API router.

### [api.py](http://_vscodecontentref_/9)

Contains the API routes for the application.

### [data_cleaning.py](http://_vscodecontentref_/10)

Handles data cleaning tasks for the dataset.

### [data_ingestion_pipeline.py](http://_vscodecontentref_/11)

Manages the data ingestion pipeline, including reading the dataset and inserting it into the database.

### [inference_pipeline.py](http://_vscodecontentref_/12)

Handles the inference pipeline for generating responses to user queries. It initializes the inference pipeline with logging and sets up the necessary configurations for OpenAI.

### [logging_classes.py](http://_vscodecontentref_/13)

Defines custom logging classes for the application.

### [config.py](http://_vscodecontentref_/14)

Contains configuration settings for the application.

### [schemas.py](http://_vscodecontentref_/15)

Defines the data schemas used in the application.

### [vector_database_ingestion_pipeline.py](http://_vscodecontentref_/16)

Manages the ingestion of data into a vector database.

## Logs

- [data_ingestion_logs.csv](http://_vscodecontentref_/17): Logs related to the data ingestion process.
- [pipeline_logs.csv](http://_vscodecontentref_/18): General logs for the data pipeline.

## Database

- `chroma_db`: Directory containing the Chroma database files.

## Environment Variables

- `HuggingFace_API_KEY`: API key for Hugging Face.
- `VoyageAI_API_KEY`: API key for Voyage AI.

## Ignored Files

- [.dockerignore](http://_vscodecontentref_/19): Specifies files to ignore when building the Docker image.
- [.gitignore](http://_vscodecontentref_/20): Specifies files to ignore in the Git repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face](https://huggingface.co/)
- [Voyage AI](https://voyageai.com/)