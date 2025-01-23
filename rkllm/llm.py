from cffi import FFI
import os
import signal
from time import sleep
from .tokens import ChatTokenizer


ffi = FFI()

PATH = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH}/rkllm.h", "r") as header:
	ffi.cdef(header.read())

rkllm = ffi.dlopen(f"{PATH}/librkllmrt.so")



class RKLLM:
	def __init__(self, model_path: str,
			  tokenizer: ChatTokenizer = None,
			  top_k = 50,
			  top_p = 0.9,
			  temperature = 0.8,
			  repeat_penalty = 1.1,
			  frequency_penalty = 0.0,
			  presence_penalty = 0.0,
			  max_new_tokens = 1024,
			  max_context_len = 122880,
			  skip_special_token = False):
		self.result = ""
		if tokenizer:
			self.tokenizer = tokenizer
		else:
			self.tokenizer = None
		self.model_path = ffi.new("char[]", model_path.encode('utf-8') + b'\0')
		
		self.default_param = rkllm.rkllm_createDefaultParam()
		self.param = ffi.new("RKLLMParam *", self.default_param)

		self.param.model_path = self.model_path

		self.param.top_k = top_k
		self.param.top_p = top_p
		self.param.temperature = temperature
		self.param.repeat_penalty = repeat_penalty
		self.param.frequency_penalty = frequency_penalty
		self.param.presence_penalty = presence_penalty

		self.param.max_new_tokens = max_new_tokens
		self.param.max_context_len = max_context_len
		self.param.skip_special_token = skip_special_token
		self.param.extend_param.base_domain_id = 0
		
		self.llmHandle = ffi.new("LLMHandle *")
		
		ret = rkllm.rkllm_init(self.llmHandle, self.param, llm_callback)
		if ret != 0:
			raise RuntimeError("Failed to load RKNN model")
		signal.signal(signal.SIGINT, self.destroy_signal)


	def chat(self, chat):
		self.result = ""
		if self.tokenizer:
			prompt = self.tokenizer.bos_token
			prompt += self.tokenizer.tokenize(chat)
		else:
			prompt += chat

		rkllm_input = ffi.new("RKLLMInput *")
		prompt_input = ffi.new("char[]", prompt.encode('utf-8') + b'\0')

		rkllm_input.input_type = rkllm.RKLLM_INPUT_PROMPT
		rkllm_input.prompt_input = prompt_input

		rkllm_infer_params = ffi.new("RKLLMInferParam *")
		rkllm_infer_params.mode = rkllm.RKLLM_INFER_GENERATE

		ret = rkllm.rkllm_run(self.llmHandle[0], rkllm_input, rkllm_infer_params, ffi.new_handle(self))
		if ret != 0:
			raise RuntimeError("Inference failed")
		return {"role": "assistant", "content": self.result}

	def destroy_signal(self, sig, frame):
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		rkllm.rkllm_abort(self.llmHandle[0])
		while rkllm.rkllm_is_running(self.llmHandle[0]) == 0:
			sleep(1)
		rkllm.rkllm_destroy(self.llmHandle[0])
		exit(0)

@ffi.callback("void(RKLLMResult *, void *, LLMCallState)")
def llm_callback(result, self, state):
	self = ffi.from_handle(self)
	if state == rkllm.RKLLM_RUN_FINISH:
		rkllm.rkllm_abort(self.llmHandle[0])
	elif state == rkllm.RKLLM_RUN_ERROR:
		raise RuntimeError("RKLLM_RUN_ERROR")
	elif state == rkllm.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
		# If last_hidden_layer has data
		if result.last_hidden_layer.embd_size != 0 and result.last_hidden_layer.num_tokens != 0:
			# Calculate the data size
			data_size = result.last_hidden_layer.embd_size * result.last_hidden_layer.num_tokens * ffi.sizeof("float")
			print(f"\ndata_size: {data_size}")
		
			# Open file to write the hidden layer data
			with open("last_hidden_layer.bin", "wb") as outFile:
				# Convert the C array (pointer) to a Python object and write to file
				outFile.write(ffi.buffer(result.last_hidden_layer.hidden_states, data_size))
				print("Data saved to last_hidden_layer.bin successfully!")
		else:
			print("No data for last_hidden_layer.")
	elif state == rkllm.RKLLM_RUN_NORMAL:
		token = ffi.string(result.text).decode('utf-8')
		
		if self.tokenizer and self.tokenizer.eos_token == token:
			rkllm.rkllm_abort(self.llmHandle[0])
		else:
			self.result += token
#		print(token, end="")


