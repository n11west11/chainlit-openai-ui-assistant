import os
import openai
from openai import NOT_GIVEN, Client

client = Client(api_key=os.environ["OPENAI_API_KEY"])


def main():
    """
    Main function to create a vector store called 
    """
    # Search for a vector store called "HTML Files"
    found_vector_store = False
    cursor = NOT_GIVEN
    while not found_vector_store:
        vector_stores = client.beta.vector_stores.list(after=cursor)
        for vs in vector_stores:
            if vs.name == "HTML Files":
                print("Found vector store called 'HTML Files'")
                found_vector_store = True
                vector_store = vs
                break
        if not found_vector_store:
            cursor = vector_store[-1].id

    if not found_vector_store:
        print("Vector store 'HTML Files' not found")
        print("Creating a new vector store called 'HTML Files'")
        vector_store = client.beta.vector_stores.create(name="HTML Files")
        print("Created vector store called 'HTML Files'")
    else:
        print("Found vector store called 'HTML Files'")
        print(f"Vector store ID: {vector_store.id}")
    return vector_store


if __name__ == "__main__":
    main()
