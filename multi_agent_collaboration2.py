from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from multi_agent_collaboration_utils import (
  create_agent,
  python_repl,
  tavily_tool,
  AgentState
)

import functools
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import ToolNode
from typing import Literal

# 辅助函数：为智能体创建一个节点
def agent_node(state, agent, name):
    # 调用智能体，获取结果
    result = agent.invoke(state)
    
    # 将智能体的输出转换为适合追加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass  # 如果是工具消息，跳过处理
    else:
        # 将结果转换为 AIMessage，并排除部分字段
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    # 返回更新后的状态，包括消息和发送者
    return {
        "messages": [result],  # 包含新生成的消息
        # 我们使用严格的工作流程，通过记录发送者来知道接下来传递给谁
        "sender": name,
    }

# 为 Agent 配置各自的大模型
#research_llm = ChatOpenAI(
#    temperature=0,
#    model="deepseek-chat",
#    openai_api_key="sk-c1ece558bdf3469099b52a63be9a6803",
#    openai_api_base="https://api.deepseek.com",
#)
#table_llm = ChatOpenAI(
#    temperature=0,
#    model="deepseek-chat",
#    openai_api_key="sk-c1ece558bdf3469099b52a63be9a6803",
#    openai_api_base="https://api.deepseek.com",
#)

research_llm = ChatOpenAI(
    temperature=0,
    model="glm-4-plus",
    openai_api_key="7b97980bf1ebf82d3a69e19d3e2adea6.NSHBqqO3luBlkb0A",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

table_llm = ChatOpenAI(
    temperature=0,
    model="glm-4-plus",
    openai_api_key="7b97980bf1ebf82d3a69e19d3e2adea6.NSHBqqO3luBlkb0A",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 研究智能体及其节点
research_agent = create_agent(
    research_llm,  # 使用 research_llm 作为研究智能体的语言模型
    [tavily_tool],  # 研究智能体使用 Tavily 搜索工具
    system_message="Before using the search engine, carefully think through and clarify the query. "
    "Then, conduct a single search that addresses all aspects of the query in one go.",  # 系统消息，指导智能体如何使用搜索工具
)

# 使用 functools.partial 创建研究智能体的节点，指定该节点的名称为 "Researcher"
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

table_agent = create_agent(
    table_llm,  # 使用 chart_llm 作为图表生成器智能体的语言模型
    [python_repl],  # 图表生成器智能体使用 Python REPL 工具
    system_message="Format a table based on the provided data.",  # 系统消息，指导智能体如何生成图表
)

# 使用 functools.partial 创建图表生成器智能体的节点，指定该节点的名称为 "Table Generator"
table_node = functools.partial(agent_node, agent=table_agent, name="Table_Generator")

# 定义工具列表，包括 Tavily 搜索工具和 Python REPL 工具
tools = [tavily_tool, python_repl]

# 创建工具节点，负责工具的调用
tool_node = ToolNode(tools)

# 创建一个状态图 workflow，使用 AgentState 来管理状态
workflow = StateGraph(AgentState)

# 将研究智能体节点、图表生成器智能体节点和工具节点添加到状态图中
workflow.add_node("Researcher", research_node)
workflow.add_node("Table_Generator", table_node)
workflow.add_node("call_tool", tool_node)

# 路由器函数，用于决定下一步是执行工具还是结束任务
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]  # 获取当前状态中的消息列表
    last_message = messages[-1]  # 获取最新的一条消息
    
    # 如果最新消息包含工具调用，则返回 "call_tool"，指示执行工具
    if last_message.tool_calls:
        return "call_tool"
    
    # 如果最新消息中包含 "FINAL ANSWER"，表示任务已完成，返回 "__end__" 结束工作流
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    
    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"

# 为 "Researcher" 智能体节点添加条件边，根据 router 函数的返回值进行分支
workflow.add_conditional_edges(
    "Researcher",
    router,  # 路由器函数决定下一步
    {
        "continue": "Table_Generator",  # 如果 router 返回 "continue"，则传递到 Table Generator
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "Chart Generator" 智能体节点添加条件边
workflow.add_conditional_edges(
    "Table_Generator",
    router,  # 同样使用 router 函数决定下一步
    {
        "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
# 工具调用节点不更新 sender 字段，这意味着边将返回给调用工具的智能体
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
    {
        "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
        "Table_Generator": "Table_Generator",  # 如果 sender 是 Table Generator，则返回给 Table Generator
    },
)

# 添加开始节点，将流程从 START 节点连接到 Researcher 节点
workflow.add_edge(START, "Researcher")

# 编译状态图以便后续使用
graph = workflow.compile()

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Obtain the GDP of the United States from 2000 to 2020, "
            "and then Format the data as a table with Python. End the task after generating the chart。"
            )
        ],
    },
    # 设置最大递归限制
    {"recursion_limit": 20},
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()  # 打印消息内容

