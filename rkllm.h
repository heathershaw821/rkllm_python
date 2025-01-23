typedef void* LLMHandle;


typedef enum {
	RKLLM_RUN_NORMAL = 0,
	RKLLM_RUN_WAITING = 1,
	RKLLM_RUN_FINISH = 2,
	RKLLM_RUN_ERROR = 3,
	RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4
} LLMCallState;





typedef enum {
	RKLLM_INPUT_PROMPT = 0,
	RKLLM_INPUT_TOKEN = 1,
	RKLLM_INPUT_EMBED = 2,
	RKLLM_INPUT_MULTIMODAL = 3,
} RKLLMInputType;





typedef enum {
	RKLLM_INFER_GENERATE = 0,
	RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1,
} RKLLMInferMode;





typedef struct {
	int32_t base_domain_id;
	uint8_t reserved[112];
} RKLLMExtendParam;





typedef struct {
	const char* model_path;
	int32_t max_context_len;
	int32_t max_new_tokens;
	int32_t top_k;
	float top_p;
	float temperature;
	float repeat_penalty;
	float frequency_penalty;
	float presence_penalty;
	int32_t mirostat;
	float mirostat_tau;
	float mirostat_eta;
	bool skip_special_token;
	bool is_async;
	const char* img_start;
	const char* img_end;
	const char* img_content;
	RKLLMExtendParam extend_param;
} RKLLMParam;





typedef struct {
	const char* lora_adapter_path;
	const char* lora_adapter_name;
	float scale;
} RKLLMLoraAdapter;





typedef struct {
	float* embed;
	size_t n_tokens;
} RKLLMEmbedInput;





typedef struct {
	int32_t* input_ids;
	size_t n_tokens;
} RKLLMTokenInput;





typedef struct {
	char* prompt;
	float* image_embed;
	size_t n_image_tokens;
} RKLLMMultiModelInput;





typedef struct {
	RKLLMInputType input_type;
	union {
		const char* prompt_input;
		RKLLMEmbedInput embed_input;
		RKLLMTokenInput token_input;
		RKLLMMultiModelInput multimodal_input;
	};
} RKLLMInput;





typedef struct {
	const char* lora_adapter_name;
} RKLLMLoraParam;





typedef struct {
	int save_prompt_cache;
	const char* prompt_cache_path;
} RKLLMPromptCacheParam;





typedef struct {
	RKLLMInferMode mode;
	RKLLMLoraParam* lora_params;
	RKLLMPromptCacheParam* prompt_cache_params;
} RKLLMInferParam;





typedef struct {
	const float* hidden_states;
	int embd_size;
	int num_tokens;
} RKLLMResultLastHiddenLayer;





typedef struct {
	const char* text;
	int32_t token_id;
	RKLLMResultLastHiddenLayer last_hidden_layer;
} RKLLMResult;

typedef void(*LLMResultCallback)(RKLLMResult* result, void* userdata, LLMCallState state);





RKLLMParam rkllm_createDefaultParam();

int rkllm_init(LLMHandle* handle, RKLLMParam* param, LLMResultCallback callback);







int rkllm_load_lora(LLMHandle handle, RKLLMLoraAdapter* lora_adapter);







int rkllm_load_prompt_cache(LLMHandle handle, const char* prompt_cache_path);






int rkllm_release_prompt_cache(LLMHandle handle);






int rkllm_destroy(LLMHandle handle);
int rkllm_run(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);
int rkllm_run_async(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);






int rkllm_abort(LLMHandle handle);






int rkllm_is_running(LLMHandle handle);
