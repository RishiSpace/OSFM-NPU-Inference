# main.py
import importlib
import sys


def detect_and_load(model_name: str):
    """
    Detect available NPU platform (Intel or AMD) and load corresponding model.
    """
    try:
        # Try Intel platform
        intel_module = importlib.import_module('intel_npu')
        tokenizer, model = intel_module.init_model(model_name)
        generate_fn = intel_module.generate_text
        print("Using Intel NPU backend.")
    except ImportError:
        try:
            # Fallback to AMD
            amd_module = importlib.import_module('amd_npu')
            tokenizer, model = amd_module.init_model(model_name)
            generate_fn = amd_module.generate_text
            print("Using AMD Ryzen AI NPU backend.")
        except ImportError:
            print("No supported NPU backend found. Please install intel_extension_for_pytorch or optimum[amd].")
            sys.exit(1)

    return tokenizer, model, generate_fn


def main():
    model_name = "meta-llama/Llama-3.2-1B" # Example model name ofc
    tokenizer, model, generate_fn = detect_and_load(model_name)

    print("NPU Inference Ready. Type 'exit' to quit.")
    while True:
        prompt = input("User: ")
        if prompt.strip().lower() in ('exit', 'quit'):
            break
        response = generate_fn(tokenizer, model, prompt)
        print(f"Assistant: {response}\n")


if __name__ == '__main__':
    main()