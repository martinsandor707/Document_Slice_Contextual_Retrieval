# Quantwise-Chunking

Chunking logic implemented according to [Anthropic's ***Contextual Retrieval***](https://www.anthropic.com/news/contextual-retrieval) flow, with limitations of at most 8 GB of VRAM in mind.

Most amount of VRAM used during chunking/embedding/etc:
***7695 MiB***

The high-level workflow looks like this:

1. The PDF's contents and metadata are extracted using Docling
2. The contents are chunked according to the token limit of the embedding model to be used later.
3. Each chunk's contents and role are summarized in relation to the entire document using a local `ollama` language model
4. The summary of each chunk is appended to the ***end*** of each chunk (See Anthropic's article about why steps 3 and 4 are needed)
5. The chunks are embedded to become dense vectors to be used for semantic similarity
6. The chunks are also used to create TF-IDF encodings which are sparse vectors used for full text search
7. All of this is saved into a vector database
8. When the database is queried, it performs a hybrid search using both semantic and full text search, returning more than the originally asked *N* entries
9. The results are fused together and reranked using a model.
10. The top *N* entries of the newly reranked chunks are returned, ready to be used for RAG.



For further reading, please consult Anthropic's excellent article on the topic.

Trivia:

There were many different AI models used to complete this retrieval process. Here is the complete list:

- Docling
    - DocLayNet (Layout Understanding)
    - TableFormer (Table extraction)
    - EasyOCR/Tesseract (In case the PDF is an image)
- AutoTokenizer
    - Based on `nomic-ai/nomic-embed-text-v1.5`
- Summarizing
    - `gemma3:4b-it-qat`
- Embedding
    - `nomic-ai/nomic-embed-text-v1.5`
- Reranking
    - `colbert-ir/colbertv2.0`

After the documents are first processed, only the embedding and Reranking models are needed for the RAG functionality, so there is plenty of room for the ollama RAG model to function.
