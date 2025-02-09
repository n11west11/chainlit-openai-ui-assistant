"""

This module contains the add_memory function, which adds a file to the assistant's memory.

 

"""

import os

from pathlib import Path

from openai import OpenAI


sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SUPPORTED_FILE_TYPES = [
    "c",
    "cpp",
    "css",
    "docx",
    "gif",
    "html",
    "java",
    "jpeg",
    "jpg",
    "js",
    "json",
    "md",
    "pdf",
    "php",
    "png",
    "pptx",
    "py",
    "rb",
    "tar",
    "tex",
    "ts",
    "txt",
    "webp",
    "xlsx",
    "xml",
    "zip",
]


async def save_information(file_name: str, extension: str, file_content: str):
    """

    Add a file to the assistant's memory

    Note this adds the file to the assistant's collective memory
    """

    # Create directory if it doesn't exist

    os.makedirs("./memories", exist_ok=True)

    # Create a temporary file to store the file content

    with open(
        "./memories/" + file_name + "." + extension, "w", encoding="utf-8"
    ) as file:

        file.write(file_content)

    
    file = Path("./memories/" + file_name + "." + extension)
    
    # Upload the file to OpenAI

    uploaded_file = sync_openai_client.files.create(
        file=file, purpose="assistants"
    )

    # Get the vector store ID attached to the assistant

    assistant = sync_openai_client.beta.assistants.retrieve(
        os.environ.get("OPENAI_ASSISTANT_ID")
    )

    # Get the vector store ID attached to the assistants file search tool

    vector_store_ids = assistant.tool_resources.file_search.vector_store_ids

    for vector_store_id in vector_store_ids:

        # Add the uploaded file to the vector store

        sync_openai_client.beta.vector_stores.files.create_and_poll(
            vector_store_id=vector_store_id, file_id=uploaded_file.id
        )

    return f"File {file_name}.{extension} has been added to the assistant's memory."

def open_ai_representation() -> dict:
    """
    Returns a dictionary representation of the function that can be used in an OpenAI assistant.
    """
    return {
        "name": "save_information",
        "description": "Add a file to the assistant's memory",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string", "description": "Name of the file"},
                "extension": {
                    "type": "string",
                    "description": f"Extension of the file (e.g., one of {SUPPORTED_FILE_TYPES})",
                },
                "file_content": {
                    "type": "string",
                    "description": "Content of the file",
                },
            },
            "required": ["file_name", "extension", "file_content"],
        },
    }
