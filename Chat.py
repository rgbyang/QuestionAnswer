# Copyright 2024-, RGBYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.vectorstores import Chroma
import EmbeddingsLocalHuggingFace
import os
from langchain.chains import RetrievalQA
import LLMHuggingFaceLocal
from langchain.prompts import PromptTemplate
import langchain
langchain.debug = True

# 加载之前生成的向量DB
model_name = "bert-base-chinese"
local_embeddings = EmbeddingsLocalHuggingFace.EmbeddingsLocalHuggingFace(model_name)
basic_DB_path = 'docs/chroma'
DB_version = "2024-08-09_23-14-54-775873"
persist_directory = os.path.join(basic_DB_path, DB_version)
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=local_embeddings
)

# 初始化链
llm = LLMHuggingFaceLocal.LLMHuggingFaceLocal()
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
# 把提示词改为中文
chinese_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="鉴于以下对话和后续问题，请将后续问题改写为一个独立问题，使用其原始语言。\n\n对话历史：{chat_history}\n后续问题：{question}\n独立问题："
)
custom_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="根据后面的上下文回答末尾的问题。如果你不知道答案，请直接说你不知道，不要试图编造一个答案。\n\n{context}\n\n问题: {question}\n答案:"
)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=chinese_prompt,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template}
)

while True:
    question = input("请输入问题：")
    if question == "exit":
        break
    fullAnswer = qa.run(question)    
    keyword = '\n答案: '
    answer = ''
    if keyword in fullAnswer: 
        answer = fullAnswer.split(keyword)[-1].strip()
    else:
        answer = fullAnswer.strip()
    print("AI助手的回答："+answer)