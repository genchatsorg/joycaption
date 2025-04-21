import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

def parse_args():
	parser = argparse.ArgumentParser(description='Generate captions using JoyCaption model')
	parser.add_argument('--model-name', type=str, default=MODEL_NAME,
					  help='Name or path of the base model')
	parser.add_argument('--lora-path', type=str, default=LORA_PATH,
					  help='Path to the LoRA weights')
    parser.add_argument('--output-path', type=str, default=OUTPUT_PATH,
					  help='Path to the output model')
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
 
    llava_model = llava_model.merge_and_unload(progressbar=True)
    llava_model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)

if __name__ == "__main__":
	main()