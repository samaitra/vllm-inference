from vllm import LLM, SamplingParams
import sys

prompts = [
    "Hello, my name is",
    "The future of AI is",
    "It is a warm sunny day",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    try:
        print("Starting vLLM inference...")
        print("Creating LLM instance...")
        
        # Create an LLM with CPU-friendly settings
        llm = LLM(
            model="facebook/opt-125m",  
            enable_prefix_caching=True, 
            tensor_parallel_size=1,  # Use 1 for CPU
            gpu_memory_utilization=0.9
        )
        
        print("LLM created successfully!")
        print("Generating text...")
        
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()