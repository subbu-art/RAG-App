import os
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI #this below has been replaced by the below import
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pinecone import Pinecone as PineconeClient

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['PINECONE_API_KEY'] = 'd9d23440-7b49-4d01-8de8-023e93367326'
os.environ['OPENAI_API_KEY'] = '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'
PC_key = os.getenv('PINECONE_API_KEY')

def data_loader(path):
    docs = PyPDFDirectoryLoader(path)
    data = docs.load()
    return data

path = 'Docs/'
data = data_loader(path)
print(len(data))

def data_splitter(docs, chunk_size = 1000, chunk_overlap = 20):
    split = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    doc_chunks = split.split_documents(docs)
    return doc_chunks

doc_chunks = data_splitter(data)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
PineconeClient(api_key = PC_key, index_api='mcq-app' )
index_name = 'mcq-app'
index = Pinecone.from_documents(doc_chunks, embeddings, index_name=index_name)

def get_similar_docs(query, k=2):
    similar_docs = index.similarity_search(query, k = k)
    return similar_docs

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
chain = load_qa_chain(llm, chain_type='stuff')

def get_answer(query):
    relevant_docs = get_similar_docs(query)
    print(relevant_docs)
    response = chain.run(input_documents = relevant_docs, question = query)
    return response

our_query = "what is indian currency?"
answer = get_answer(our_query)
print(answer)


response_schemas = [
    ResponseSchema(name="question", description="Question generated from provided input text data."),
    ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
    ResponseSchema(name="answer", description="Correct answer for the asked question.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

chat_model = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""When a text input is given by the user, please generate multiple choice questions 
        from it along with the correct answer. 
        \n{format_instructions}\n{user_prompt}""")  
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

final_query = prompt.format_prompt(user_prompt = answer)
print(final_query)
final_query.to_messages()
final_query_output = chat_model.invoke(final_query.to_messages())
print(final_query_output.content)
