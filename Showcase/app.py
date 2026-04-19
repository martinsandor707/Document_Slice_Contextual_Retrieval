import json
import os

import chainlit as cl
import lancedb
import ollama
import torch
from chainlit.input_widget import Select, Switch
from lancedb.embeddings import get_registry
from lancedb.rerankers import ColbertReranker

# --- Configs ---
DB_PATH = "./db"
CONTEXTUAL_TABLE_NAME = os.environ.get("TABLE_NAME", "my_anthropic_sliding_table")
TRADITIONAL_TABLE_NAME = "semantic_table"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
# RAG_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma3:4b-it-qat")
RAG_CHAT_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
# --- Prompt Templates ---
ROUTER_PROMPT = """
You are an intelligent classification system.
User Query: "{question}"

Your task: Determine if this query requires looking up information in a medical/scientific knowledge base (dentistry, diabetes, health, documents), or if it is a general conversational greeting/question (like "hello", "who are you", "what is the capital of belgium").

Output ONLY one word:
- "RAG" (if it needs external knowledge from documents)
- "GENERAL" (if it is general chat or world knowledge)
"""

RAG_PROMPT_TEMPLATE = """
You are a helpful expert assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, politely say that you don't have information about that in the documents. 
CRITICAL: ONLY ANSWER IN ENGLISH.  

Context:
{context}

Question: {question}
"""


class DbHandler:
    def __init__(self, db_path, embedding_model_name):
        self.db = lancedb.connect(db_path)
        self.reranker = ColbertReranker()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = (
            get_registry()
            .get("huggingface")
            .create(name=embedding_model_name, trust_remote_code=True, device=device)
        )

    def get_table_names(self):
        """Returns a list of table names present in the LanceDB instance."""
        try:
            return self.db.table_names()
        except Exception as e:
            print(f"Error fetching table names: {e}")
            return []

    def query_table(self, prompt, use_contextual_rag, limit=3):
        try:
            table_name = (
                CONTEXTUAL_TABLE_NAME if use_contextual_rag else TRADITIONAL_TABLE_NAME
            )
            table = self.db.open_table(table_name)
        except Exception:
            return []

        if use_contextual_rag:
            results_df = (
                table.search(
                    prompt,
                    query_type="hybrid",
                    vector_column_name="vector",
                    fts_columns="text",
                )
                .rerank(reranker=self.reranker)
                .limit(limit)
                .to_pandas()
            )
        else:
            results_df = (
                table.search(
                    prompt,
                    query_type="vector",
                )
                .limit(limit)
                .to_pandas()
            )

        return (
            results_df["original_text"].tolist(),
            results_df["id"].tolist(),
            results_df["document"].tolist(),
        )


# --- Global variables ---
db_handler = None
ollama_client = None


@cl.on_chat_start
async def start():
    global db_handler, ollama_client

    if db_handler is None:
        try:
            db_handler = DbHandler(
                db_path=DB_PATH,
                embedding_model_name=EMBEDDING_MODEL_NAME,
            )
        except Exception:
            pass

    if db_handler:
        available_tables = db_handler.get_table_names()

    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_cot",
                label="Show chain of thought (Debug)",
                initial=True,
            ),
            Switch(
                id="use_contextual_rag",
                label="Use contextual RAG",
                values=available_tables,
                initial=True,
            ),
        ]
    ).send()

    ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)

    await cl.Message(content="👋 Hello! I'm ready to answer your questions!").send()

    cl.user_session.set(
        "starters",
        [
            cl.Starter(
                label="Fogágybetegség",
                message="Milyen tünetei vannak a fogágybetegségnek?",
                icon="/public/idea.svg",
            ),
            cl.Starter(
                label="Diabetes",
                message="Mi a különbség az I-es és II-es típusú cukorbetegség között?",
                icon="/public/learn.svg",
            ),
            cl.Starter(
                label="Teszt (Semleges)",
                message="Mi Belgium fővárosa?",
                icon="/public/terminal.svg",
            ),
        ],
    )


@cl.on_settings_update
async def setup_agent(settings):
    pass


@cl.on_message
async def main(message: cl.Message):
    user_query = message.content

    settings = cl.user_session.get("chat_settings", {})
    show_cot = settings.get("show_cot", True)
    use_contextual_rag = settings.get("use_contextual_rag", True)

    # --- Step 1: ROUTING (Not shown, only in background) ---
    router_decision = "RAG"

    try:
        router_response = await ollama_client.generate(
            model=RAG_CHAT_MODEL,
            prompt=ROUTER_PROMPT.format(question=user_query),
        )
        decision_text = router_response["response"].strip().upper()
        if "GENERAL" in decision_text:
            router_decision = "GENERAL"
    except Exception as e:
        print("Routing failed, defaulting to RAG.")
        print(e)
        pass

    final_prompt = user_query
    context_str = ""

    # --- Step 2: Branching based on router ---
    if router_decision == "RAG":

        async def perform_search():
            if db_handler:
                return await cl.make_async(db_handler.query_table)(
                    user_query, use_contextual_rag, limit=3
                )
            return []

        context_chunks = []

        if show_cot:
            async with cl.Step(
                name=f"Knowledge Base Search (using {'Contextual RAG' if use_contextual_rag else 'Traditional RAG'})",
                type="tool",
            ) as step:
                step.input = user_query
                context_chunks, chunk_ids, source_documents = await perform_search()

                if context_chunks:
                    details = []
                    eval_message = ""

                    with open("example_questions.json", "r", encoding="utf-8") as f:
                        examples = json.load(f)

                    # Find matching question in examples
                    matching_example = next(
                        (ex for ex in examples if ex["question"] == user_query), None
                    )
                    found_count = 0
                    if matching_example:
                        supporting_chunks = matching_example.get(
                            "supporting_chunks", []
                        )
                        expected_document = matching_example.get("document", "N/A")
                        expected_answer = matching_example.get("gold_answer", "N/A")
                        found_count = 0
                        for i, hash in enumerate(chunk_ids):
                            if hash in supporting_chunks:
                                details.append(
                                    f"✅ Chunk {i + 1} is a bullseye: ...{context_chunks[i][:150]}...\nSource Document: {source_documents[i][:50]}..."
                                )
                                found_count += 1
                            else:
                                details.append(
                                    f"❌ Chunk {i + 1} is a miss: ...{context_chunks[i][:150]}...\nSource Document: {source_documents[i][:50]}..."
                                )
                    eval_message = f"📊 **Evaluation:** Found {found_count} out of {len(supporting_chunks)} expected supporting chunks."

                    details = "\n\n".join(details)
                    step.output = f"Question recognized from dataset! Expected document: {expected_document[:50]}...\n\n📌 Top **{len(context_chunks)} chunks:**\n{details}\n\n{eval_message}\n\n🏆 Expected 'Golden' answer:\n{expected_answer}"
                    context_str = "\n\n---\n\n".join(context_chunks)
                else:
                    context_str = "No relevant entry in the database."
                    step.output = "❌ No relevant information found in the documents."
        else:
            context_chunks = await perform_search()
            if context_chunks:
                context_str = "\n\n---\n\n".join(context_chunks)
            else:
                context_str = "No relevant entry in the database."

        final_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=user_query,
        )

    else:
        if show_cot:
            async with cl.Step(
                name="General conversation",
                type="llm",
            ) as step:
                step.output = (
                    "The question is general in nature, "
                    "no need to search the knowledge base."
                )

        final_prompt = user_query

    # --- Step 3: Generating Answer ---
    msg = cl.Message(content="")
    await msg.send()

    try:
        stream = await ollama_client.chat(
            model=RAG_CHAT_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            stream=True,
        )

        async for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                await msg.stream_token(chunk["message"]["content"])
    except Exception as e:
        msg.content = f"❌ Error: {e}"
        await msg.update()

    await msg.update()

