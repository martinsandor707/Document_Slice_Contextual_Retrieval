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

## Source papers excluded for licensing reasons

> **This private archive is complete.** Everything described below as removed from the *public* repository (the five papers' PDFs, chunk files, generated contexts and database rows) is present here in its original form, so that the archive's history stays consistent with the runs reported in the publications. Do not make this repository public without repeating the removal.

The benchmark corpus of the article consists of 25 open-access scientific papers. A licence check of every PDF (Crossref metadata, October 2026) showed that five of them may not be redistributed, or not in derived form: four are published under licences that permit reading but not republication, and one is CC BY-NC-ND, which permits sharing the unmodified article but not derivative works such as cleaned text chunks and machine-generated summaries. On the advice of the project supervisor, every copy of those papers' text was removed from the public repository on 2026-10-14. The other 20 papers are CC BY licensed and remain in full.

| Paper | Venue, year | DOI | Licence | Removed from the public repository |
|---|---|---|---|---|
| Machine behaviour (Rahwan et al.) | Nature, 2019 | [10.1038/s41586-019-1138-y](https://doi.org/10.1038/s41586-019-1138-y) | subscription article (Springer Nature text-and-data-mining licence only) | PDF, chunk files, generated contexts, database rows |
| Electron Paramagnetic Resonance Study on 28Si Single Crystal for the Future Realization of the Kilogram | IEEE Transactions on Instrumentation and Measurement, 2019 | [10.1109/TIM.2018.2884044](https://doi.org/10.1109/TIM.2018.2884044) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| Stock Market Prediction via Multi-Source Multiple Instance Learning | IEEE Access, 2018 | [10.1109/ACCESS.2018.2869735](https://doi.org/10.1109/ACCESS.2018.2869735) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| The Graph Database: Jack of All Trades or Just Not SQL? | IT Professional, 2017 | [10.1109/MITP.2017.4241475](https://doi.org/10.1109/MITP.2017.4241475) | subscription article (© IEEE) | PDF, chunk files, generated contexts, database rows |
| The Application of the SOFM Neural Network and Internet of Things in Rural Revitalization | IEEE Access, 2023 | [10.1109/ACCESS.2023.3327131](https://doi.org/10.1109/ACCESS.2023.3327131) | CC BY-NC-ND 4.0 (unmodified copies may be shared, derivatives may not) | chunk files, generated contexts, database rows (the unmodified PDF stays in `RnD/input/`) |

What this means in practice:

- `RnD/input/` holds 21 of the 25 PDFs; `RnD/split_documents/`, `RnD/split_documents/AutoTokenizer/` and `RnD/split_documents_end_to_end/` hold the chunk files of 20 papers; every file under `RnD/preprocessed_chunks/`, the run records under `RnD/dynamic_slice_length/router_experiments/real_runs/`, the gold labels in `RnD/dynamic_slice_length/slice_radius_gold.json`, the older question set `RnD/q_and_a/scientific.json`, and every LanceDB table under `RnD/db/` and `Showcase/db/` contain the records of those 20 papers only (292 of the original 382 chunks).
- Everything that carries no article text is complete: the question sets in `RnD/q_and_a/Gemini/` (they reference chunks by content hash), the per-question retrieval results, the routing records, the energy ledger and the notebooks.
- All numbers, tables and figures in the article were computed on the full 25-paper corpus, so the verification scripts in `RnD/verification/` reproduce the per-corpus statistics only after the missing records are restored.
- To reproduce the experiments end to end, download the five PDFs through the DOIs above (institutional access is needed for the two subscription articles), place them in `RnD/input/` under their original file names, and run `RnD/end_to_end_dynamic_doc_slice_generation.ipynb`, which parses, routes and summarises them again; summaries will not be bit-identical to the archived run. The original chunk and context records of the five papers are available from the authors on reasonable request for verification purposes.

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
