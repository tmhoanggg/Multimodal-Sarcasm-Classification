from torch.utils.data import Dataset
import os
from PIL import Image
import json


class MyDataset(Dataset):
    def __init__(self, args, mode, limit=None):
        self.args = args
        self.data = self.load_data(args, mode, limit)
        self.image_ids = list(self.data.keys())  # Use unique indices as keys
        for id in self.data.keys():
            if mode in ["train"]:
                self.data[id]["image_path"] = os.path.join(self.args.image_train, self.data[id]["image"])
            else:
                self.data[id]["image_path"] = os.path.join(self.args.image_test, self.data[id]["image"])
    
    def load_data(self, args, mode, limit=None):
        cnt = 0
        data_set = {}
        label_mapping = {
            "not-sarcasm": 0,
            "multi-sarcasm": 1,
            "text-sarcasm": 2,
            "image-sarcasm": 3
        }
        
        if mode in ["train"]:
            with open(self.args.text_train, 'r', encoding='utf-8') as f:
                datas = json.load(f)
                for key, data in datas.items():
                    if limit is not None and cnt >= limit:
                        break

                    file_name = data['image']
                    sentence = data['caption']
                    label = label_mapping[data['label']]
                    
                    cur_img_path = os.path.join(self.args.image_train, file_name)
                    if not os.path.exists(cur_img_path):
                        print(f"{cur_img_path} not found!")
                        continue
                    
                    data_set[key] = {
                        "image": file_name,
                        "caption": sentence,
                        "label": label
                    }
                    cnt += 1
                    
        elif mode in ["test"]:
            with open(self.args.text_test, 'r', encoding='utf-8') as f:
                datas = json.load(f)
                for key, data in datas.items():
                    file_name = data['image']
                    sentence = data['caption']
                    label = data['label']

                    cur_img_path = os.path.join(self.args.image_test, file_name)
                    if not os.path.exists(cur_img_path):
                        print(f"{cur_img_path} not found!")
                        continue
                    
                    data_set[key] = {
                        "image": file_name,
                        "caption": sentence,
                        "label": label
                    }
                    cnt += 1
                    
        else:
            print("Not found correct mode in MyDataset class!!!")
        
        return data_set

    def image_loader(self, id):
        return Image.open(self.data[id]["image_path"])

    def text_loader(self, id):
        return self.data[id]["caption"]

    def __getitem__(self, index):
        id = self.image_ids[index]  # Access by unique key (index from JSON)
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text, image_feature, label, id

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list
