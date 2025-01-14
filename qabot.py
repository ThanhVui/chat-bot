from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from sympy.polys.polyconfig import query

# Configuration
model_file = 'models/vinallama-7b-chat_q5_0.gguf'
db_path = 'vectorstores/db_faiss'
# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Create prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

# Create simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k':3}, max_tokens=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt':prompt}
    )
    return llm_chain

# Read from vector DB
def read_from_vector_db():
    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(folder_path=db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db

# Run
db = read_from_vector_db()
llm = load_llm(model_file)

# Template
template = """<lim_start|>system\nsử dụng thông tin sau đầy để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)

# Create LLM chain
llm_chain = create_qa_chain(prompt, llm, db)

# Run
query = "Phân loại BA và ưu/nhược điểm cho tôi?"
response = llm_chain.invoke({"query": query})
print(response)