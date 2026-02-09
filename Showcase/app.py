import os
import chainlit as cl
import lancedb
from lancedb.embeddings import get_registry
from lancedb.rerankers import ColbertReranker
import torch
import ollama
from chainlit.input_widget import Switch

# --- Konfiguráció ---
DB_PATH = "./db"
TABLE_NAME = os.environ.get("TABLE_NAME", "my_table")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
RAG_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma3:4b-it-qat")

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

Context:
{context}

Question: {question}
"""

# --- Adatbázis Kezelő ---
class DbHandler:
    def __init__(self, db_path, embedding_model_name):
        self.db = lancedb.connect(db_path)
        self.reranker = ColbertReranker()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = get_registry().get("huggingface").create(
            name=embedding_model_name,
            trust_remote_code=True,
            device=device
        )

    def query_table(self, table_name, prompt, limit=3):
        try:
            table = self.db.open_table(table_name)
        except Exception:
            return []

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
        return results_df["text"].tolist()

# --- Globális változók ---
db_handler = None
ollama_client = None

@cl.on_chat_start
async def start():
    global db_handler, ollama_client

    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_cot",
                label="Gondolatmenet mutatása (Debug)",
                initial=True,
            ),
        ]
    ).send()

    ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)

    if db_handler is None:
        try:
            db_handler = DbHandler(
                db_path=DB_PATH,
                embedding_model_name=EMBEDDING_MODEL_NAME,
            )
        except Exception:
            pass

    await cl.Message(
        content="👋 Hello! I'm ready to answer your questions!"
    ).send()

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

    settings = cl.user_session.get("chat_settings", {"show_cot": True})
    show_cot = settings["show_cot"]

    # --- 1. LÉPÉS: ROUTING (TELJESEN HÁTTÉRBEN, SOHA NEM JELENIK MEG) ---
    router_decision = "RAG"

    try:
        router_response = await ollama_client.generate(
            model=RAG_CHAT_MODEL,
            prompt=ROUTER_PROMPT.format(question=user_query),
        )
        decision_text = router_response["response"].strip().upper()
        if "GENERAL" in decision_text:
            router_decision = "GENERAL"
    except Exception:
        pass

    final_prompt = user_query
    context_str = ""

    # --- 2. LÉPÉS: Ágválasztás ---
    if router_decision == "RAG":

        async def perform_search():
            if db_handler:
                return await cl.make_async(
                    db_handler.query_table
                )(TABLE_NAME, user_query, limit=3)
            return []

        context_chunks = []

        if show_cot:
            async with cl.Step(
                name="Keresés a Tudásbázisban",
                type="tool",
            ) as step:
                step.input = user_query
                context_chunks = await perform_search()

                if context_chunks:
                    details = "\n\n".join(
                        [
                            f"📄 **Találat {i+1}:** ...{chunk[:150]}..."
                            for i, chunk in enumerate(context_chunks)
                        ]
                    )
                    step.output = f"✅ **{len(context_chunks)} találat:**\n{details}"
                    context_str = "\n\n---\n\n".join(context_chunks)
                else:
                    context_str = "Nincs releváns találat az adatbázisban."
                    step.output = "❌ Nem találtam releváns információt a dokumentumokban."
        else:
            context_chunks = await perform_search()
            if context_chunks:
                context_str = "\n\n---\n\n".join(context_chunks)
            else:
                context_str = "Nincs releváns találat az adatbázisban."

        final_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=user_query,
        )

    else:
        if show_cot:
            async with cl.Step(
                name="Általános csevegés",
                type="llm",
            ) as step:
                step.output = (
                    "A kérdés általános jellegű, "
                    "nem szükséges a tudásbázisban keresni."
                )

        final_prompt = user_query

    # --- 3. LÉPÉS: Válaszgenerálás ---
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
        msg.content = f"❌ Hiba: {e}"
        await msg.update()

    await msg.update()