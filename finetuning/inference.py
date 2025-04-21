"""
HF_HOME=/workspace/cache python inference.py --image data/downloaded_images/a21ce500-d54c-424b-a59f-3d212a593fcf-IMG_7857.jpg --lora-path checkpoints/hegqdlp2/samples_576/model
"""

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


IMAGE_PATH = "image.jpg"
PROMPT = "Write a list of tags for this image."
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
LORA_PATH = "../joy-caption-finetune/checkpoints/cuu2y0sx/samples_1984/model"


def parse_args():
	parser = argparse.ArgumentParser(description='Generate captions using JoyCaption model')
	parser.add_argument('--image', type=str, default="image.jpg",
					  help='Path to the input image')
	parser.add_argument('--prompt', type=str, default=PROMPT,
					  help='Prompt for the captioning model')
	parser.add_argument('--model-name', type=str, default=MODEL_NAME,
					  help='Name or path of the base model')
	parser.add_argument('--lora-path', type=str, default=LORA_PATH,
					  help='Path to the LoRA weights')
	parser.add_argument('--temperature', type=float, default=0.6,
					  help='Sampling temperature (default: 0.6)')
	parser.add_argument('--top-p', type=float, default=0.9,
					  help='Top-p sampling parameter (default: 0.9)')
	parser.add_argument('--max-tokens', type=int, default=300,
					  help='Maximum number of tokens to generate (default: 300)')
	return parser.parse_args()

def main():
	args = parse_args()
	
	# Load JoyCaption
	# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
	# device_map=0 loads the model into the first GPU
	processor = AutoProcessor.from_pretrained(args.model_name)
	llava_model = LlavaForConditionalGeneration.from_pretrained(args.model_name, torch_dtype="bfloat16", device_map=0)
	llava_model = PeftModel.from_pretrained(llava_model, args.lora_path)
	llava_model.eval()

	with torch.no_grad():
		# Load image
		image = Image.open(args.image)

		# Build the conversation
		convo = [
			{
				"role": "system",
				"content": "You are a helpful image captioner.",
			},
			{
				"role": "user",
				"content": args.prompt,
			},
		]

		# Format the conversation
		# WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
		# but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
		# if not careful, which can make the model perform poorly.
		convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
		assert isinstance(convo_string, str)

		# Process the inputs
		inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
		inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

		# Generate the captions
		generate_ids = llava_model.generate(
			**inputs,
			max_new_tokens=args.max_tokens,
			do_sample=True,
			suppress_tokens=None,
			use_cache=True,
			temperature=args.temperature,
			top_k=None,
			top_p=args.top_p,
		)[0]

		# Trim off the prompt
		generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

		# Decode the caption
		caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
		caption = caption.strip()
		print(caption)

if __name__ == "__main__":
	main()