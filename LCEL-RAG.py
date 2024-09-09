from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

with open("letter.txt", "r", encoding="utf-8") as f:
    letters = f.read()
    
# 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.create_documents([letters])

# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
)

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    openai_api_key="sk-c1ece558bdf3469099b52a63be9a6803",
    openai_api_base="https://api.deepseek.com",
)



# 自定义提示词模板
template = """根据以下上下文来回答问题。如果你不知道答案，就直接说你不知道，不要试图编造答案。答案最多三句话，简洁明了。结尾记得说“谢谢提问！”

{context}

问题: {question}

答案:"""

custom_rag_prompt = PromptTemplate.from_template(template)

custom_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# 使用自定义 prompt 生成回答
print(custom_rag_chain.invoke("这封信中讨论了几个更新?")) #这封信中讨论了四个更新：安全和隐私措施、人力资源更新和员工福利、营销举措和活动、研发项目。谢谢提问！
print(custom_rag_chain.invoke("迈克尔·约翰逊负责哪一块业务?")) #迈克尔·约翰逊负责人力资源更新和员工福利相关业务。谢谢提问！
print(custom_rag_chain.invoke("头脑风暴会议是几号?")) #头脑风暴会议定于7月10日举行。谢谢提问！
