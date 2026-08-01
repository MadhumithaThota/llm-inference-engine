import torch

from engine.model_loader import load_model, load_tokenizer

# Load model and tokenizer
tokenizer = load_tokenizer()
model = load_model()

prompt = "What is Machine Learning?"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True,
    )

cache = outputs.past_key_values

print("=" * 80)
print("CACHE TYPE")
print("=" * 80)
print(type(cache))

print("\nCACHE ATTRIBUTES")
print("=" * 80)
print(dir(cache))

print("\nNUMBER OF LAYERS")
print("=" * 80)

try:
    print(len(cache))
except Exception as e:
    print("Cannot get length:", e)

if hasattr(cache, "layers"):

    print("\nNumber of layers:", len(cache.layers))

    print("\nFIRST LAYER TYPE")
    print("=" * 80)
    first_layer = cache.layers[0]
    print(type(first_layer))

    print("\nFIRST LAYER ATTRIBUTES")
    print("=" * 80)
    print(dir(first_layer))

    print("\nFIRST LAYER REPRESENTATION")
    print("=" * 80)
    print(first_layer)

    print("\nSEARCHING FOR TENSORS")
    print("=" * 80)

    for attr in dir(first_layer):

        if attr.startswith("_"):
            continue

        try:
            value = getattr(first_layer, attr)

            if torch.is_tensor(value):
                print(f"{attr}: Tensor {value.shape}")

            elif isinstance(value, (list, tuple)):
                print(f"{attr}: {type(value).__name__} (length={len(value)})")

            else:
                print(f"{attr}: {type(value)}")

        except Exception:
            pass