import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# embeddings = OpenAIEmbeddings(
#     model='text-embedding-3-small',
#     show_progress_bar=False,
#     chunk_size=50,
#     retry_min_seconds=10
# )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100,          # texts per API request
    max_retries=10,          # default is low
    retry_min_seconds=5,
    retry_max_seconds=60,
)


vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(
#     index_name="langchain-doc-index", embedding=embeddings
# )
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], batch_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), batch_size):
        chunk = urls[i : i + batch_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f"🔄 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs.",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(
            input={"urls": urls, "extract_depth": "advanced"}
        )
        extracted_docs_count = len(docs.get("results", []))
        if extracted_docs_count > 0:
            log_success(
                f"TavilyExtract: Completed batch {batch_num} - extracted {extracted_docs_count} documents"
            )
        else:
            log_error(
                f"TavilyExtract: Batch {batch_num} failed to extract any documents, {docs}"
            )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return {'results': []}

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"🔧 TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages

# async def index_documents_async(documents: List[Document], batch_size: int = 50):
#     """Process documents in batches asynchronously."""
#     log_header("VECTOR STORAGE PHASE")
#     log_info(
#         f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
#         Colors.DARKCYAN,
#     )

#     # Create batches
#     batches = [
#         documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
#     ]

#     log_info(
#         f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
#     )

#     # Process all batches concurrently
#     async def add_batch(batch: List[Document], batch_num: int):
#         try:
#             await vectorstore.aadd_documents(batch)
#             log_success(
#                 f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
#             )
#         except Exception as e:
#             log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
#             return False
#         return True

#     # Process batches concurrently
#     tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)

#     # Count successful batches
#     successful = sum(1 for result in results if result is True)

#     if successful == len(batches):
#         log_success(
#             f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
#         )
#     else:
#         log_warning(
#             f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
#         )

import hashlib

def doc_id(d):  # deterministic → retries/re-runs don't duplicate
    return hashlib.md5(f"{d.metadata['source']}|{d.page_content}".encode()).hexdigest()

async def index_documents_async(documents, batch_size=200, concurrency=2):
    batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
    sem = asyncio.Semaphore(concurrency)

    async def add_batch(batch, n):
        async with sem:
            for attempt in range(5):
                try:
                    await vectorstore.aadd_documents(batch, ids=[doc_id(d) for d in batch])
                    log_success(f"Batch {n}/{len(batches)} ok")
                    return True
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        wait = 2 ** attempt * 5
                        log_warning(f"Batch {n} rate-limited, retry in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        log_error(f"Batch {n} failed: {e}")
                        return False

    results = await asyncio.gather(*(add_batch(b, i+1) for i, b in enumerate(batches)))
    ok = sum(results)
    log_info(f"Indexed {ok}/{len(batches)} batches")
    return ok == len(batches)

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyMap: Starting to crawl documentation from https://docs.langchain.com/oss/python/",
        Colors.PURPLE,
    )

    # res = tavily_crawl.invoke({
    #     'url':'https://docs.langchain.com/oss/python/',
    #     'max_depth':1,
    #     'limit': 100,
    #     'extract_depth':'advanced',
    #     #'instructions':'content on ai agent'
    # })
    
    # results = res.get("results", []) if isinstance(res, dict) else []
    # if not results:
    #     log_error(f"Crawl returned nothing: {res}")

    # all_docs = [
    #     Document(page_content=r["raw_content"], metadata={"source": r["url"]})
    #     for r in results
    #     if r.get("raw_content")
    # ]
    # # all_docs = [Document(page_content=result['raw_content'], metadata={'source': result['url']}) for result in res['results']]
    
    # log_success(
    #     f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    # )

    site_map = tavily_map.invoke("https://docs.langchain.com/oss/python/")
    log_success(
        f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site"
    )

    # Split URLs into batches of 20
    url_batches = chunk_urls(list(site_map["results"]), batch_size=20)
    log_info(
        f"📋 URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    # await index_documents_async(splitted_docs, batch_size=500)
    ok = await index_documents_async(splitted_docs)
    if ok:
        log_success("Pipeline finished successfully")
    else:
        log_warning("Pipeline finished with failures")

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(site_map['results'])}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == '__main__':
    asyncio.run(main())