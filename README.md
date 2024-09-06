## Overview
This is a LLM Question and Answer system. You can ask questions for a specified context. Supports multiple rounds of questioning and answering. Supports Chinese and English. Using the langchain methods to load context documents and split them. Using the torch to embed context. Using the Chroma of langchain to generate and load the vector DB. Using the langchain.memory and the langchain ConversationalRetrievalChain to support multiple rounds of questioning and answering. Using the transformers model from huggingface.


## Getting Started
1. Put the expected context *.txt files to folder "docs\Context"
2. Run Vectorization.py, will store the embedding result to a new folder under "docs\chroma", remember it.
3. Set the new folder name to DB_version in Chat.py, then run Chat.py, then enter the question in the CMD terminal.
4. After it output the answer, you can enter the new question. The old question and answer will be remembered as the history of the conversation.

## License

This project is licensed under the Apache License 2.0. 

### Third-Party Licenses

This project uses the following third-party libraries:

- **Library transformers**: Licensed under the Apache License 2.0
- **Library torch**: Licensed under the BSD License
- **Library langchain**: Licensed under the MIT License

For more details, see the [LICENSE](LICENSE) file.