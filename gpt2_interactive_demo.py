import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# Page configuration
st.set_page_config(
    page_title="🤖 GPT-2 Text Generator",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .generated-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load GPT-2 model and tokenizer"""
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(prompt, max_length, temperature, top_p, num_return_sequences):
    """Generate text using GPT-2"""
    model, tokenizer = load_model()
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with st.spinner('📝 Generating text...'):
        start_time = time.time()
        
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the original prompt from the generated text
        generated_text = text[len(prompt):].strip()
        results.append(generated_text)
    
    return results, generation_time

def main():
    """Main Streamlit app"""
    st.title("🤖 GPT-2 Text Generation Demo")
    st.markdown("### Professional AI Text Generator")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Generation Parameters")
    
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=50,
        max_value=500,
        value=150,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Controls randomness (lower = more focused, higher = more creative)"
    )
    
    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Controls diversity by sampling from top tokens"
    )
    
    num_sequences = st.sidebar.slider(
        "Number of Variations",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Generate multiple text variations"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Enter Your Prompt")
        prompt = st.text_area(
            "Prompt",
            height=100,
            placeholder="Enter your text prompt here...",
            help="The starting text for GPT-2 to continue"
        )
        
        # Preset prompts
        preset_prompts = {
            "🚀 Future of AI": "The future of artificial intelligence will",
            "🎭 Creative Story": "Once upon a time in a world where",
            "💼 Business": "The key to successful entrepreneurship is",
            "🧪 Science": "Recent breakthroughs in quantum computing have",
            "🎵 Music": "The evolution of music through technology shows"
        }
        
        selected_preset = st.selectbox(
            "Quick Prompts",
            ["Custom"] + list(preset_prompts.keys())
        )
        
        if selected_preset != "Custom":
            prompt = preset_prompts[selected_preset]
    
    with col2:
        st.markdown("### 📊 Generation Stats")
        stats_placeholder = st.empty()
    
    # Generate button
    if st.button("🚀 Generate Text", type="primary"):
        if prompt:
            results, gen_time = generate_text(
                prompt, max_length, temperature, top_p, num_sequences
            )
            
            # Display stats
            stats_placeholder.markdown(f"""
            - **Generation Time:** {gen_time:.2f}s
            - **Tokens Generated:** ~{len(results[0].split())} words
            - **Model:** GPT-2 (117M parameters)
            """)
            
            # Display results
            st.markdown("### 🎯 Generated Text Variations")
            
            for i, result in enumerate(results, 1):
                st.markdown(f"**Variation {i}:**")
                st.markdown(f'<div class="generated-text">{prompt}{result}</div>', unsafe_allow_html=True)
                
                # Copy button
                st.code(f"{prompt}{result}", language=None)
                
        else:
            st.warning("⚠️ Please enter a prompt to generate text.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 Tips for Better Results:**
    - **Lower temperature (0.1-0.5)**: More focused and deterministic
    - **Higher temperature (0.8-1.2)**: More creative and diverse
    - **Top-p**: Lower values (0.7-0.9) for more coherent text
    """)

if __name__ == "__main__":
    main()
