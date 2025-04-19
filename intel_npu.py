import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import intel_extension_for_pytorch as ipex


def init_model(model_name: str, dtype=torch.float16, device: str = "xpu"):
    """
    Initialize and optimize a Hugging Face causal LM for Intel NPU execution.

    Args:
        model_name: Name of the model on Hugging Face (e.g., "meta-llama/Llama-3.2-1B").
        dtype: Torch dtype for model parameters (default: torch.float16).
        device: Device string for Intel NPU (default: "xpu").

    Returns:
        tokenizer: AutoTokenizer instance for the model.
        model: Optimized and compiled PyTorch model on specified device.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply Intel Extension for PyTorch optimizations
    model = ipex.optimize(model, dtype=dtype)
    model.to(device)

    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 50) -> str:
    """
    Generate text with the optimized model.

    Args:
        tokenizer: Tokenizer returned from init_model.
        model: Model returned from init_model.
        prompt: Input text prompt.
        max_new_tokens: Number of tokens to generate (default: 50).

    Returns:
        Generated text string.
    """
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)