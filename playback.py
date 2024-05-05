from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
import os
from llama_index.core import Settings


llm = Ollama(model="mistral", request_timeout=30)
Settings.llm = llm
embed_model = resolve_embed_model("local:BAAI/bge-m3")
Settings.embed_model = embed_model
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage/")

# load index
os.environ["OPENAI_API_KEY"] = "random"
new_index = load_index_from_storage(storage_context)
new_query_engine = new_index.as_query_engine(llm=llm)
res = new_query_engine.query("Is sungwon expert in AI and deep learning field?")
print(res)

