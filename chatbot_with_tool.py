from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from utils import BasicToolNode
from langchain_core.messages import BaseMessage
from typing import Literal
import os

'''
对比结果如下：
Assistant: [{"url": "https://www.curotec.com/insights/langchain-vs-langgraph-framework-comparison/", "content": "Opt for LangGraph if your project involves understanding and visualizing complex data relationships. LangGraph excels in areas like recommendation systems, knowledge graphs, and social network analysis, where the connections between data points are crucial. Its graph-based modeling and visualization tools enable you to map and analyze intricate ..."}, {"url": "https://medium.com/@lucas.dahan/hands-on-langgraph-building-a-multi-agent-assistant-06aa68ed942f", "content": "With a simple python main.py command, your first LangGraph project is up and running. This version explicitly describes the choices made by the graph. The idea is to provide insights to help you ..."}]
Assistant: LangGraph 项目是一个专注于理解和可视化复杂数据关系的框架。它特别适用于需要分析数据点之间连接的项目，如推荐系 统、知识图谱和社交网络分析。LangGraph 的图模型和可视化工具能够帮助用户映射和分析复杂的数据关系。

例如，在推荐系统中，LangGraph 可以帮助理解用户和产品之间的关系，从而提供更精准的推荐。在知识图谱中，它可以用于构建和可视化实体之间的关系，帮助用户更好地理解和利用知识。在社交网络分析中，LangGraph 可以揭示用户之间的互动模式，帮助企业优化社交策略。

此外，LangGraph 还提供了一个简单易用的 Python 接口，用户可以通过运行 `python main.py` 命令快速启动他们的第一个 LangGraph 项目。这个框架的设计旨在提供清晰的见解，帮助用户在项目中做出明智的选择。

总的来说，LangGraph 是一个强大的工具，适用于需要深入分析和可视化复杂数据关系的各种项目。
'''




# 定义 Tavily 搜索工具，最大搜索结果数设置为 2
tool = TavilySearchResults(max_results=2)
tools = [tool]

# 测试工具调用
tool.invoke("What's a 'node' in LangGraph?")

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 初始化 LLM 并绑定搜索工具
chat_model = ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    openai_api_key="sk-c1ece558bdf3469099b52a63be9a6803",
    openai_api_base="https://api.deepseek.com",
)
llm_with_tools = chat_model.bind_tools(tools)

# 更新聊天机器人节点函数，支持工具调用
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 将更新后的节点添加到状态图中
graph_builder.add_node("chatbot", chatbot)

# 将 BasicToolNode 添加到状态图中
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 定义路由函数，检查工具调用
def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    使用条件边来检查最后一条消息中是否有工具调用。
    
    参数:
    state: 状态字典或消息列表，用于存储当前对话的状态和消息。
    
    返回:
    如果最后一条消息包含工具调用，返回 "tools" 节点，表示需要执行工具调用；
    否则返回 "__end__"，表示直接结束流程。
    """
    # 检查状态是否是列表类型（即消息列表），取最后一条 AI 消息
    if isinstance(state, list):
        ai_message = state[-1]
    # 否则从状态字典中获取 "messages" 键，取最后一条消息
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    # 如果没有找到消息，则抛出异常
    else:
        raise ValueError(f"输入状态中未找到消息: {state}")

    # 检查最后一条消息是否有工具调用请求
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"  # 如果有工具调用请求，返回 "tools" 节点
    return "__end__"  # 否则返回 "__end__"，流程结束

# 添加条件边，判断是否需要调用工具
graph_builder.add_conditional_edges(
    "chatbot",  # 从聊天机器人节点开始
    route_tools,  # 路由函数，决定下一个节点
    {"tools": "tools", "__end__": "__end__"},  # 定义条件的输出，工具调用走 "tools"，否则走 "__end__"
)

# 当工具调用完成后，返回到聊天机器人节点以继续对话
graph_builder.add_edge("tools", "chatbot")

# 指定从 START 节点开始，进入聊天机器人节点
graph_builder.add_edge(START, "chatbot")

# 编译状态图，生成可执行的流程图
graph = graph_builder.compile()

# 进入一个无限循环，用于模拟持续的对话
while True:
    # 获取用户输入
    user_input = input("User: ")
    
    # 如果用户输入 "quit"、"exit" 或 "q"，则退出循环，结束对话
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")  # 打印告别语
        break  # 退出循环

    # 使用 graph.stream 处理用户输入，并生成机器人的回复
    # "messages" 列表中包含用户的输入，传递给对话系统
    for event in graph.stream({"messages": [("user", user_input)]}):
        
        # 遍历 event 的所有值，检查是否是 BaseMessage 类型的消息
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                # 如果消息是 BaseMessage 类型，则打印机器人的回复
                print("Assistant:", value["messages"][-1].content)