# 导入相关模块，包括运算符、输出解析器、聊天模板、ChatOpenAI 和 运行器
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    openai_api_key="sk-c1ece558bdf3469099b52a63be9a6803",
    openai_api_base="https://api.deepseek.com",
)

# 创建一个计划器，生成一个关于给定输入的论证
planner = (
    ChatPromptTemplate.from_template("{input}")
    | llm
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

# 创建正面论证的处理链，列出关于基础回应的正面或有利的方面
arguments_for = (
    ChatPromptTemplate.from_template(
        "使用python语言实现{base_response}这一功能"
    )
    | llm
    | StrOutputParser()
)

# 创建反面论证的处理链，列出关于基础回应的反面或不利的方面
arguments_against = (
    ChatPromptTemplate.from_template(
        "使用Java语言实现{base_response}这一功能"
    )
    | llm
    | StrOutputParser()
)

# 创建最终响应者，综合原始回应和正反论点生成最终的回应
final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "python代码:\n{results_1}\n\nJava代码:\n{results_2}"),
            ("system", "无需做过多解释，直接将两段代码分别输出。"),
        ]
    )
    | llm
    | StrOutputParser()
)

# 构建完整的处理链，从生成论点到列出正反论点，再到生成最终回应
chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

print(chain.invoke({"input": "二分法查找"}))