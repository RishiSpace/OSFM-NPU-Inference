from transformers import AutoTokenizer
from optimum.amd.ryzenai import RyzenAIModel


def init_model(model_name: str):
    """
    Initialize an AMD Ryzen AI optimized model for NPU execution.

    Args:
        model_name: Name of the model on Hugging Face (e.g., "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix").

    Returns:
        tokenizer: AutoTokenizer instance for the model.
        session: ONNX Runtime InferenceSession for the model.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load or export ONNX model via RyzenAIModel
    ryzen_model = RyzenAIModel.from_pretrained(model_name, export=True)
    session: ort.InferenceSession = ryzen_model.model

    return tokenizer, session


def generate_text(tokenizer, session, prompt: str, max_new_tokens: int = 50) -> str:
    """
    Generate text using the ONNX Runtime session on AMD NPU.

    Args:
        tokenizer: Tokenizer returned from init_model.
        session: ONNX Runtime InferenceSession returned from init_model.
        prompt: Input text prompt.
        max_new_tokens: Number of tokens to generate (default: 50).

    Returns:
        Generated text string.
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Convert to numpy
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

    # Run inference
    outputs = session.run(None, ort_inputs)[0]

    return tokenizer.decode(outputs[0], skip_special_tokens=True)