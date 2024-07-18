import gradio as gr
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str = "db_data"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]

    if len(ans["source_documents"]) == 0:
        template = """
            以下是之前的对话：
            {previous_dialogue}
            客户的问题是：{question}
            请重新给一个更自然、连贯的回复，要像一个顶级的私人教练一样
            """
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = PromptTemplate(template=template, input_variables=["previous_dialogue", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(previous_dialogue=history, question=message)

        return response
    # 否则输出套路话术
    else:
        return "对不起，我需要查找一下资料"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="私教机器人",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    import os

    os.environ["OPENAI_API_KEY"] = 'sk-0RDQ2Nikw7RCuaq9Ee5d6493DcA645559eE277C177874443'
    os.environ["OPENAI_BASE_URL"] = 'https://api.xiaoai.plus/v1'

    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
