# adding prompt caching and batching

from vllm import LLM, SamplingParams

# Initialize the LLM engine with prefix caching enabled
llm = LLM(model="facebook/opt-125m", enable_prefix_caching=True)

# Define prompts and sampling parameters
prompts = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy cat",
    "A different prompt entirely"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")