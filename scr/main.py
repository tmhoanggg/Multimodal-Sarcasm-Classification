import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor, AutoTokenizer, AutoProcessor
import pickle
from PIL import ImageFile
from sklearn.model_selection import train_test_split
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
from train import train 
from model import MV_CLIP
from data_set import MyDataset


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def compute_class_weights(args, train_data):
    # Count the number of occurrences of each class
    class_counts = torch.zeros(args.label_number)
    
    for data in train_data:
        _, _, label, _ = data
        class_counts[label] += 1
    
    # Compute class weights
    total_samples = class_counts.sum().item()
    class_weights = total_samples / (class_counts * len(class_counts))

    return class_weights
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for Multimodal Sarcasm Detection")

    parser.add_argument('--device', type=str, default='0', help="Device to run the model on (e.g., '0' for GPU, 'cpu')")
    parser.add_argument('--model', type=str, default='MV_CLIP', help="Model name to use")
    parser.add_argument('--text_train', type=str, default='D:/Contest/DSC/data/old/merged_train_en.json', help="Path to text training data")
    parser.add_argument('--image_train', type=str, default='D:/Contest/DSC/data/old/aug_images_final/train-images', help="Path to image training data")
    parser.add_argument('--simple_linear', type=bool, default=False, help="Use a simple linear model")
    parser.add_argument('--num_train_epochs', type=int, default=4, help="Number of training epochs")
    parser.add_argument('--train_batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--label_number', type=int, default=4, help="Number of labels")
    parser.add_argument('--text_size', type=int, default=512, help="Text size")
    parser.add_argument('--image_size', type=int, default=768, help="Image size")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help="Epsilon value for Adam optimizer")
    parser.add_argument('--optimizer_name', type=str, default='adam', help="Optimizer name")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--clip_learning_rate', type=float, default=1e-6, help="Learning rate for CLIP")
    parser.add_argument('--max_len', type=int, default=77, help="Maximum sequence length for text")
    parser.add_argument('--layers', type=int, default=6, help="Number of layers")
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help="Maximum gradient norm")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay rate")
    parser.add_argument('--warmup_proportion', type=float, default=0.2, help="Warmup proportion for scheduler")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/', help="Directory to save outputs")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of samples")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--model_path', type=str, default='/kaggle/working/MV_CLIP', help="Path to save or load the model")
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help="CLIP model name")
    parser.add_argument('--current_epoch', type=int, default=9, help="Current epoch (for resuming training)")
    parser.add_argument('--attempt', type=str, default='', help="Additional attempt description")

    return parser.parse_args()
        

def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)

    # Load full training data và test data
    train_data = MyDataset(args, mode='train', limit=None)
    
    # Compute class weights
    class_weights = compute_class_weights(args, train_data)
    class_weights = class_weights.to(device)  # Move to same device as model
    
    if args.model == 'MV_CLIP':
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        model = MV_CLIP(args, class_weights=class_weights)
    else:
        raise RuntimeError('Error model name!')

    #model.load_state_dict(torch.load('/kaggle/working/MV_CLIP/model4.pt', map_location="cpu"))
    model.to(device)

    train(args, model, device, train_data, processor)

if __name__ == '__main__':
    main() 