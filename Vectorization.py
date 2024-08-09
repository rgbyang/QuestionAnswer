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
from langchain.document_loaders import TextLoader #PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import EmbeddingsLocalHuggingFace
import os
from datetime import datetime

# 读取指定文件夹里的所有txt文档作为上下文
loaders = []
for root, dirs, files in os.walk("docs/Context"):
    for file in files:
        if file.endswith(".txt"):
            loaders.append(TextLoader(file_path=os.path.join(root, file), encoding="utf-8"))
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 分割文档
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["(?<=\。 )",""]
)
splits = r_splitter.split_documents(docs)

# 用本地HuggingFace模型初始化Embeddings
model_name = "bert-base-chinese"
local_embeddings = EmbeddingsLocalHuggingFace.EmbeddingsLocalHuggingFace(model_name)

# 使用 Chroma 创建嵌入索引
# 每次把DB生成到新的目录中，目录名为'docs/chroma/2024-08-08_16-10-45-545601'
basic_DB_path = 'docs/chroma'
now = datetime.now()
currentTime = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
persist_directory = os.path.join(basic_DB_path, currentTime)
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
chroma = Chroma.from_documents(
    documents=splits,
    embedding=local_embeddings,
    persist_directory=persist_directory
    )