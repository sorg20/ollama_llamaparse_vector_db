from langchain.llms import HuggingFaceEndpoint
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
import os
from llama_index.core import Settings

HF_TOKEN = "XXX"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
llm = HuggingFaceEndpoint(
        endpoint_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=512,
        repetition_penalty= 1.1,
        temperature= 0.5,
        top_p= 0.9,
        return_full_text=False
        )
Settings.llm = llm
load_dotenv()
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

#embed_model = resolve_embed_model("local:BAAI/bge-m3")
embed_model = resolve_embed_model("local:BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

res = query_engine.query("which univeristy Sungwon gradudate?")
print(res)
vector_index.storage_context.persist(persist_dir="./storage/")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage/")

# load index
os.environ["OPENAI_API_KEY"] = "random"
new_index = load_index_from_storage(storage_context)
new_query_engine = new_index.as_query_engine(llm=llm)
res = new_query_engine.query("What is sungwon's hobby?")
print(res)

