import os
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import json
import numpy as np
from zipfile import ZipFile
import argparse
from model import MV_CLIP
from data_set import MyDataset


def predict(args, model, device, data, processor, pre = None):
    data_loader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=MyDataset.collate_func,shuffle=False)
    results = {}  # Sử dụng dict để lưu kết quả dự đoán
    index = 0 # Để lưu id của file kết quả

    model.eval()
    with open(pre,'w',encoding='utf-8') as fout:
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, _, id_list = t_batch  # Nhận các phần tử từ batch, bỏ qua label
                
                # Xử lý đầu vào cho model
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                
                # Dự đoán đầu ra
                t_outputs = model(inputs, labels=None)
                predict = torch.argmax(t_outputs[0], -1).cpu().numpy().tolist()
                
                for pred in predict:
                    results[index] = ['not-sarcasm', 'multi-sarcasm', 'text-sarcasm', 'image-sarcasm'][pred]
                    index += 1
                
    # Save predictions to JSON and compress into a zip file
    with ZipFile(pre, 'w') as zipf:
        with zipf.open('results.json', 'w') as json_file:
            json_data = json.dumps({"results": results, "phase": "dev"}, ensure_ascii=False)
            json_file.write(json_data.encode('utf-8'))
    
    print("Predictions have been saved to", pre)      

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for Multimodal Sarcasm Detection - Testing Phase")

    parser.add_argument('--device', type=str, default='0', help="Device to run the model on (e.g., '0' for GPU, 'cpu')")
    parser.add_argument('--max_len', type=int, default=77, help="Maximum sequence length for text")
    parser.add_argument('--text_size', type=int, default=512, help="Text size")
    parser.add_argument('--image_size', type=int, default=768, help="Image size")
    parser.add_argument('--dropout_rate', type=float, default=0, help="Dropout rate")
    parser.add_argument('--label_number', type=int, default=4, help="Number of labels")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size for testing")
    parser.add_argument('--model_path', type=str, default='/kaggle/working/MV_CLIP', help="Path to the trained model")
    parser.add_argument('--save_file', type=str, default='B32_HAdata_processed_en_epoch4.zip', help="Name of the saved file")
    parser.add_argument('--text_test', type=str, default='/kaggle/input/text-processed-en/vimmsd-test-processed-en.json', help="Path to text test data")
    parser.add_argument('--image_test', type=str, default='/kaggle/input/muiltimodal-sarcasm/data/public-test-images', help="Path to image test data")
    parser.add_argument('--layers', type=int, default=6, help="Number of layers")
    parser.add_argument('--simple_linear', type=bool, default=False, help="Use a simple linear model")
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help="CLIP model name")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = MV_CLIP(args)

    test_data = MyDataset(args, mode='test', limit=None)

    model.load_state_dict(torch.load('/kaggle/working/MV_CLIP/model4.pt', map_location="cpu"), strict=False)
    model.to(device)
    model.eval()

    predict(args, model, device, test_data, processor, pre=args.save_file)