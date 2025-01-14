from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Declare variables
pdf_data_path = 'data'
vetor_db_path = 'vectorstores/db_faiss'


# Function 1: Create vector DB from one pharagraph text
def create_db_from_text():
    raw_text = """BA Là cầu nối giữa các bên liên quan, đảm bảo hiểu rõ yêu cầu và đề xuất giải pháp đáp ứng
mục tiêu kinh doanh. Công việc bao gồm thu thập, phân tích, tài liệu hóa, giao tiếp, và quản lý yêu cầu"""

    # Split content
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

    # Take it into file Faiss DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vetor_db_path)
    return db

def create_db_from_file():
    # Load data from file PDF
    loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    # Hash data
    chunks = text_splitter.split_documents(documents)
    # Extract text content from the Document objects
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Embedding model
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_texts(chunk_texts, embedding_model)
    db.save_local(vetor_db_path)
    return db


# create_db_from_file()
create_db_from_text()
# if __name__ == "__main__":
#     # db = create_db_from_text()
#     # db = create_db_from_file()
#     print(db)