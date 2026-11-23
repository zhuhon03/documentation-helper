import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embedding)

# Initialize chat model
model = init_chat_model(model="gpt-5-nano", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""

    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
        - answer: The generated answer
        - context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    # system_prompt = (
    #     "You are a helpful AI assistant that answers questions about LangChain documentation. "
    #     "You have access to a tool that retrieves relevant documentation. "
    #     "Use the tool to find relevant information before answering questions. "
    #     "Always cite the sources you use in your answers. "
    #     "If you cannot find the answer in the retrieved documentation, say so."
    # )

    system_prompt = """
    You answer questions about LangChain using its official documentation.

    Rules:
    - Always call the `retrieve` tool before answering any LangChain question, even if you think you know the answer. APIs change often; your prior knowledge may be outdated.
    - If results look irrelevant or incomplete, retry once with a rephrased, more specific query.
    - Answer only from retrieved content. Do not fill gaps with prior knowledge.
    - Cite inline as [1], [2] matching the doc ids, and end with a "Sources:" list of the URLs actually used. Never invent URLs.
    - If the docs don't cover the question, say so plainly and state what you did find.
    - For code, prefer examples shown in the docs; keep imports exactly as documented.
    - Off-topic questions: briefly say you only cover LangChain docs.

    Be concise.
    """

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

    # Build message list
    messages =[{'role':'user', 'content':query}]

    # Invoke the agent
    response = agent.invoke({'messages':messages})

    # Extract the answer from the last AI message
    answer = response['messages'][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query='What are deep agents?')
    print(result)