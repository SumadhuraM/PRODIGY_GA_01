from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad token to eos token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.8, top_p=0.9):
    """Generate text using GPT-2 with enhanced parameters"""
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create attention mask
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    
    # Generate with better parameters
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def interactive_demo():
    """Interactive text generation demo"""
    print("🤖 GPT-2 Text Generation Demo")
    print("=" * 50)
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        try:
            length = int(input("Enter max length (default 100): ") or "100")
            temp = float(input("Enter temperature (default 0.8): ") or "0.8")
            
            print("\n📝 Generating...")
            result = generate_text(prompt, length, temp)
            print(f"\n🎯 Generated Text:\n{result}")
            
        except ValueError:
            print("❌ Please enter valid numbers")

if __name__ == "__main__":
    # Quick test
    print("Testing GPT-2 generation...")
    test_result = generate_text("The future of AI is", max_length=50)
    print(f"Test result: {test_result}")
    
    # Uncomment for interactive mode
    # interactive_demo()
