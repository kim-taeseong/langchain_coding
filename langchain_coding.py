from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=openai_api_key)

# output = llm.invoke('2024년 1월 경제 뉴스에 대해서 알려줘')

# print(output)

prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 취업 및 채용지원에 대해서 알려주는 로봇이야'),
    ('user', '{input}')
])

chain = prompt | llm

# output = chain.invoke({'input': '국민취업지원제도 중 2유형의 취업활동비용은 얼마야?'})

# print(output)

loader = WebBaseLoader('https://www.moel.go.kr/policy/policyinfo/support/list4.do')

docs = loader.load()

embeddings = OpenAIEmbeddings(openai_api_type=openai_api_key)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''
    다음의 구조를 기반으로 질문에 답해
    <context>
    {context}
    </context>

질문: {input}
''')

document_chain = create_stuff_documents_chain(llm, prompt)

# result = document_chain.invoke({
#     'input': '국민취업지원제도가 뭐야',
#     'context': [Document(page_content='''국민취업지원제도란?

# 취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
# [출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24''')]
# })

# print(result)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({'input': '국민취업지원제도 중 2유형의 취업활동비용은 얼마야?'})
response = retrieval_chain.invoke({'input': '지원대상이 어떻게 돼?'})
print(response['answer'])
