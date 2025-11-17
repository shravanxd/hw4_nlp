from load_data import T5Dataset
from transformers import T5TokenizerFast

# Test with and without schema enhancement
print("=" * 80)
print("Testing input format WITHOUT schema enhancement:")
print("=" * 80)
dataset_no_schema = T5Dataset('data', 'train', use_schema_enhancement=False)
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# Get first example
encoder_input, decoder_input, decoder_target = dataset_no_schema[0]
encoder_text = tokenizer.decode(encoder_input, skip_special_tokens=True)
decoder_input_text = tokenizer.decode(decoder_input, skip_special_tokens=True)
decoder_target_text = tokenizer.decode(decoder_target, skip_special_tokens=True)

print(f"\nExample 1 (no schema):")
print(f"Encoder input ({len(encoder_input)} tokens): {encoder_text}")
print(f"Decoder input ({len(decoder_input)} tokens): {decoder_input_text}")
print(f"Decoder target ({len(decoder_target)} tokens): {decoder_target_text}")

print("\n" + "=" * 80)
print("Testing input format WITH schema enhancement:")
print("=" * 80)
dataset_with_schema = T5Dataset('data', 'train', use_schema_enhancement=True)

# Get first example
encoder_input, decoder_input, decoder_target = dataset_with_schema[0]
encoder_text = tokenizer.decode(encoder_input, skip_special_tokens=True)
decoder_input_text = tokenizer.decode(decoder_input, skip_special_tokens=True)
decoder_target_text = tokenizer.decode(decoder_target, skip_special_tokens=True)

print(f"\nExample 1 (with schema):")
print(f"Encoder input ({len(encoder_input)} tokens):")
print(f"  First 200 chars: {encoder_text[:200]}...")
print(f"Decoder input ({len(decoder_input)} tokens): {decoder_input_text}")
print(f"Decoder target ({len(decoder_target)} tokens): {decoder_target_text}")

# Check if END token is in decoder target
print(f"\nEND token in decoder target: {'END' in decoder_target_text}")

# Show a few more examples
print("\n" + "=" * 80)
print("Additional examples (no schema):")
print("=" * 80)
for i in range(1, 4):
    encoder_input, decoder_input, decoder_target = dataset_no_schema[i]
    encoder_text = tokenizer.decode(encoder_input, skip_special_tokens=True)
    decoder_target_text = tokenizer.decode(decoder_target, skip_special_tokens=True)
    print(f"\nExample {i+1}:")
    print(f"  Encoder: {encoder_text}")
    print(f"  Target: {decoder_target_text}")
    print(f"  Has END: {'END' in decoder_target_text}")
