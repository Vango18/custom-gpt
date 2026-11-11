"""
RAG chatbot over your own PDF / DOCX files.

Works with either an OpenAI key or a free Google Gemini key (https://aistudio.google.com/apikey).

Run:
    pip install streamlit langchain-openai langchain-google-genai langchain-pinecone langchain-community langchain-text-splitters pinecone pymupdf docx2txt
    streamlit run rag_chatbot_app.py
"""

import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

EMBED_DIM = 1024  # both providers output 1024-dim vectors, so the Pinecone setup is identical
MEMORY_TURNS = 5  # simple memory: remember the last 5 question/answer pairs

# Both Gemini models below are on Google's free tier.
PROVIDERS = {
    "OpenAI": {
        "id": "openai",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1-mini",
        "key_label": "OpenAI API key",
        "key_help": "Get one at https://platform.openai.com/api-keys",
    },
    "Google Gemini (free)": {
        "id": "gemini",
        "embed_model": "gemini-embedding-2",
        "chat_model": "gemini-3.8-flash",  # if you hit free-tier limits, try "gemini-3.5-flash-lite"
        "key_label": "Gemini API key",
        "key_help": "Free key at https://aistudio.google.com/apikey",
    },
}


def make_embeddings(provider: dict, api_key: str):
    if provider["id"] == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=provider["embed_model"], google_api_key=api_key, output_dimensionality=EMBED_DIM
        )
    return OpenAIEmbeddings(model=provider["embed_model"], dimensions=EMBED_DIM, api_key=api_key)


def make_llm(provider: dict, api_key: str):
    if provider["id"] == "gemini":
        return ChatGoogleGenerativeAI(model=provider["chat_model"], temperature=0, google_api_key=api_key)
    return ChatOpenAI(model=provider["chat_model"], temperature=0, api_key=api_key)

st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- session state
defaults = {
    "indexed": False,     # True once the index is ready to chat with
    "steps": [],          # log of indexing steps shown in the sidebar
    "messages": [],       # chat history: list of {"role", "content"}
    "index_name": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def log_step(text: str, placeholder):
    """Add an indexing step to the sidebar log and redraw it."""
    st.session_state.steps.append(text)
    placeholder.markdown("\n\n".join(st.session_state.steps))


# ---------------------------------------------------------------- sidebar: setup
with st.sidebar:
    st.header("⚙️ Setup")
    provider_name = st.radio(
        "LLM provider",
        list(PROVIDERS),
        horizontal=True,
        help="No OpenAI key? Pick Gemini and use a free Google AI Studio key.",
    )
    provider = PROVIDERS[provider_name]
    llm_key = st.text_input(provider["key_label"], type="password", help=provider["key_help"])
    st.caption(f"Embeddings: `{provider['embed_model']}` · Chat: `{provider['chat_model']}`")
    pinecone_key = st.text_input("Pinecone API key", type="password")
    index_name = st.text_input(
        "Pinecone index name",
        placeholder="e.g. my-docs",
        help="Lowercase letters, numbers and hyphens only. Created if it does not exist.",
    ).strip().lower()

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX files", type=["pdf", "docx"], accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    index_btn = col1.button("📥 Index files", use_container_width=True, type="primary")
    connect_btn = col2.button(
        "🔌 Use existing",
        use_container_width=True,
        help="Skip uploading and chat with an index you already filled before.",
    )

    st.divider()
    st.subheader("📋 Indexing steps")
    steps_box = st.empty()
    steps_box.markdown("\n\n".join(st.session_state.steps) or "_Nothing indexed yet._")


def keys_ok() -> bool:
    if not (llm_key and pinecone_key and index_name):
        st.sidebar.error("Please enter both API keys and an index name.")
        return False
    return True


def load_file(uploaded) -> list:
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    try:
        loader = PyMuPDFLoader(path) if suffix == ".pdf" else Docx2txtLoader(path)
        docs = loader.load()
    finally:
        os.remove(path)
    for d in docs:  # show the real file name instead of the temp path
        d.metadata["source"] = uploaded.name
    return docs


def ensure_index(pc: Pinecone, placeholder) -> bool:
    if pc.has_index(index_name):
        dim = pc.describe_index(index_name).dimension
        if dim != EMBED_DIM:
            log_step(f"❌ Index **{index_name}** has dimension {dim}, expected {EMBED_DIM}.", placeholder)
            return False
        # OpenAI and Gemini vectors live in different spaces — an index can only be searched
        # with the provider that filled it. Untagged indexes were created by the OpenAI-only version.
        tags = pc.describe_index(index_name).tags or {}
        built_with = tags.get("embed_provider", "openai")
        if built_with != provider["id"]:
            log_step(
                f"❌ Index **{index_name}** was built with **{built_with}** embeddings, but you selected "
                f"**{provider['id']}**. Switch provider or use a different index name.",
                placeholder,
            )
            return False
        log_step(f"✅ Using existing index **{index_name}** ({built_with} embeddings)", placeholder)
    else:
        log_step(f"⏳ Creating index **{index_name}** (this can take ~30s)...", placeholder)
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags={"embed_provider": provider["id"]},
        )
        log_step(f"✅ Index **{index_name}** created", placeholder)
    return True


def file_already_indexed(index, file_name: str) -> bool:
    """True if the index already holds at least one chunk from this file name.

    Uses a metadata filter on "source", so it also works for indexes filled before this check
    existed. The query vector is a dummy one: only the filter matters, so no embedding call is spent.
    """
    res = index.query(vector=[1.0] * EMBED_DIM, top_k=1, filter={"source": {"$eq": file_name}})
    return bool(res.matches)


# ---------------------------------------------------------------- indexing
if index_btn and keys_ok():
    if not uploaded_files:
        st.sidebar.error("Please upload at least one file.")
    else:
        st.session_state.steps = []
        st.session_state.indexed = False
        try:
            with st.sidebar:
                with st.spinner("Indexing..."):
                    # Step 1: pinecone index (checked first, so duplicates are caught before any work)
                    pc = Pinecone(api_key=pinecone_key)
                    if not ensure_index(pc, steps_box):
                        raise ValueError("This index can't be used with the selected provider — see the step above.")
                    index = pc.Index(index_name)
                    log_step("✅ **Step 1 – Index ready**", steps_box)

                    # Step 2: duplicate check — skip files whose name is already in this index
                    new_files, seen = [], set()
                    for f in uploaded_files:
                        if f.name in seen:
                            continue  # same file uploaded twice in this batch
                        seen.add(f.name)
                        if file_already_indexed(index, f.name):
                            log_step(f"⏭️ **{f.name}** is already in **{index_name}** — skipped", steps_box)
                        else:
                            new_files.append(f)
                    skipped = len(seen) - len(new_files)
                    log_step(
                        f"✅ **Step 2 – Duplicate check:** {len(new_files)} new, {skipped} already indexed",
                        steps_box,
                    )

                    if new_files:
                        # Step 3: load
                        docs = []
                        for f in new_files:
                            file_docs = load_file(f)
                            docs.extend(file_docs)
                            log_step(f"📄 Loaded **{f.name}** — {len(file_docs)} page(s)", steps_box)
                        log_step(f"✅ **Step 3 – Load:** {len(docs)} page(s) from {len(new_files)} file(s)", steps_box)

                        # Step 4: split
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
                        log_step(f"✅ **Step 4 – Split:** {len(chunks)} chunks (size 1000, overlap 200)", steps_box)
                        if not chunks:
                            raise ValueError("No text found in the uploaded files (scanned PDFs need OCR).")

                        # Step 5: embed + insert (same chunk -> same id, so a retry overwrites instead of duplicating)
                        embedding = make_embeddings(provider, llm_key)
                        ids = [
                            hashlib.md5(
                                f"{c.metadata.get('source')}|{c.metadata.get('page')}|{c.page_content}".encode()
                            ).hexdigest()
                            for c in chunks
                        ]
                        store = PineconeVectorStore(index=index, embedding=embedding)
                        store.add_documents(chunks, ids=ids)
                        log_step(
                            f"✅ **Step 5 – Embed + insert:** {len(ids)} vectors with `{provider['embed_model']}` "
                            f"({EMBED_DIM} dims)",
                            steps_box,
                        )

            if new_files:
                log_step("🎉 **Indexing complete — start chatting!**", steps_box)
            else:
                log_step(
                    "🎉 **All files are already indexed — nothing to do. Connected, start chatting!**", steps_box
                )
            st.session_state.indexed = True
            st.session_state.index_name = index_name
            st.session_state.provider_name = provider_name
            st.session_state.messages = []
        except Exception as e:
            log_step(f"❌ **Error:** {e}", steps_box)

if connect_btn and keys_ok():
    st.session_state.steps = []
    try:
        pc = Pinecone(api_key=pinecone_key)
        if not pc.has_index(index_name):
            log_step(f"❌ Index **{index_name}** does not exist. Upload files to create it.", steps_box)
        elif ensure_index(pc, steps_box):
            count = pc.Index(index_name).describe_index_stats().get("total_vector_count", 0)
            log_step(f"✅ Connected — {count} vectors in the index", steps_box)
            st.session_state.indexed = True
            st.session_state.index_name = index_name
            st.session_state.provider_name = provider_name
            st.session_state.messages = []
    except Exception as e:
        log_step(f"❌ **Error:** {e}", steps_box)

# ---------------------------------------------------------------- main area
st.title("📄 Chat with your documents")

if not st.session_state.indexed:
    st.info(
        "👈 Pick a provider (**OpenAI** or free **Gemini**), enter its API key, your **Pinecone** key and an "
        "**index name** in the sidebar, upload PDF/DOCX files and click **Index files**. "
        "The chatbot appears here once indexing finishes."
    )
    st.stop()

if st.session_state.get("provider_name") != provider_name:
    st.warning(
        f"This index was opened with **{st.session_state.get('provider_name')}**. Switch the provider back, "
        "or click Index files / Use existing again."
    )
    st.stop()

if not (llm_key and pinecone_key):
    st.warning("API keys were cleared — please enter them again in the sidebar.")
    st.stop()


@st.cache_resource(show_spinner=False)
def build_chain(provider_name: str, llm_key: str, pinecone_key: str, index_name: str):
    """Build retriever + LLM chains once per (provider, keys, index)."""
    provider = PROVIDERS[provider_name]
    embedding = make_embeddings(provider, llm_key)
    pc = Pinecone(api_key=pinecone_key)
    retriever = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding).as_retriever(
        search_kwargs={"k": 4}
    )
    llm = make_llm(provider, llm_key)

    # 1) Rewrite a follow-up ("what about its timeline?") into a standalone question for retrieval
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite the question so it can be "
         "understood without the history. Do NOT answer it — only return the rewritten question. "
         "If it is already standalone, return it unchanged."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 2) Answer using retrieved context + chat history
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions about the user's documents. "
         "Use only the context below and the conversation so far. "
         "If the answer is not in the context, say \"I don't know\".\n\nContext:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()
    return retriever, rewrite_chain, answer_chain


retriever, rewrite_chain, answer_chain = build_chain(
    provider_name, llm_key, pinecone_key, st.session_state.index_name
)


def get_history():
    """Simple window memory: last MEMORY_TURNS question/answer pairs as LangChain messages."""
    recent = st.session_state.messages[-MEMORY_TURNS * 2:]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
    ]


top = st.columns([4, 1])
top[0].caption(
    f"Connected to index **{st.session_state.index_name}** · {provider_name} (`{provider['chat_model']}`) "
    f"· memory: last {MEMORY_TURNS} turns"
)
if top[1].button("🧹 Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.markdown(m["sources"])

if question := st.chat_input("Ask something about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)

    history = get_history()
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                standalone = rewrite_chain.invoke({"history": history, "question": question}) if history else question
                docs = retriever.invoke(standalone)
                context = "\n\n".join(d.page_content for d in docs)

            answer = st.write_stream(
                answer_chain.stream({"context": context, "history": history, "question": question})
            )
            sources = "\n".join(
                f"- **{d.metadata.get('source', '?')}**"
                + (f" — page {int(d.metadata['page']) + 1}" if d.metadata.get("page") is not None else "")
                + f": {d.page_content[:150].strip()}..."
                for d in docs
            )
            with st.expander("Sources"):
                if standalone != question:
                    st.caption(f"Searched for: _{standalone}_")
                st.markdown(sources or "_No chunks retrieved._")
        except Exception as e:
            answer, sources = f"⚠️ Error: {e}", ""
            st.error(answer)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
