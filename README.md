# rkllm_python
RKLLM didn’t come with Python bindings, leaving developers hanging and making it a pain to integrate. This was *INSANE* given the current AI ecosystem. So, I said screw that and built these bindings myself, tearing down the barriers and making RKLLM dead simple for Python devs to use. Now you can harness the raw power of the Rockchip NPU for NLP tasks with almost zero hassle. Oh, and did I mention? There's Hugging Face tokenizer integration baked in, so you can easily work with models like the ones I converted here: [https://huggingface.co/heathershaw](https://huggingface.co/heathershaw) using my maintained fork of ez-er-rklm-toolkit here: [https://github.com/heathershaw821/ez-er-rkllm-toolkit2](https://github.com/heathershaw821/ez-er-rkllm-toolkit2).

**❤️ You're welcome ❤️**


## How to use it

```python
from rkllm.llm import RKLLM
from rkllm.tokens import ChatTokenizer

model_path = "./Llama-3.2-3B-Instruct-chatml-rk3588-1.1.2/Llama-3.2-3B/Llama-3.2-3B.rkllm"
tokenizer = ChatTokenizer("./Llama-3.2-3B-Instruct-chatml-rk3588-1.1.2/tokenizer_config.json")

rknn = RKLLM(model_path,
          tokenizer=tokenizer
          top_k = 40,
          top_p = 0.85,
          temperature = 0.7,
          repeat_penalty = 1.2,
          frequency_penalty = 0.2,
          presence_penalty = 0.6,
          max_new_tokens = 1024,
          max_context_len = 122880)

chat = [{"role": "system", "content": """
You are a helpful, knowledgeable, and friendly AI assistant.
Answer questions accurately and concisely, and provide clear explanations when necessary.
If you don't know the answer, admit it rather than guessing."""}]

while True:
  message = input(">> ")
  if message == "exit":
    exit(0)
  else:
    chat.append({"role": "user", "content": message})

  response = rknn.chat(chat)

  print(response)
  chat.append(response)

```
