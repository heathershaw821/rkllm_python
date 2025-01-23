from jinja2 import Template
import json

class ChatTokenizer:
	def __init__(self, template=None):
		if template:
			self.load_template(template)
		else:
			self.bos_token = "<|begin_of_text|>"
			self.eos_token = "<|eot_id|>"
			self.template = Template("{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}")

	def tokenize(self, chat):
		tokens = self.template.render(messages=chat, add_generation_prompt=True)
		#print(tokens)
		return tokens

	def load_template(self, template):
		with open(template, "r") as file:
			config = json.load(file)
		self.bos_token = config['bos_token']
		self.eos_token = config['eos_token']
		self.template = Template(config['chat_template'])
		#print(self.bos_token, self.eos_token, self.template)

