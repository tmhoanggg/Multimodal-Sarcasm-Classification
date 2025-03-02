import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm, trange
from sklearn import metrics
import numpy as np
from data_set import MyDataset


# def train(args, model, device, train_data, dev_data, processor):
def train(args, model, device, train_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=MyDataset.collate_func,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    
    model.to(device)

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'MV_CLIP':
            clip_params = list(map(id, model.model.parameters()))
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
            optimizer = AdamW([
                    {"params": base_params},
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate}
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        else:
            optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            if args.model == 'MV_CLIP':
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        print(f"Epoch {i_epoch + 1}")
        print(f"Train loss {sum_loss/sum_step}")

        
        path_to_save = os.path.join(args.output_dir, args.model)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        model_to_save = (model.module if hasattr(model, "module") else model)
        torch.save(model_to_save.state_dict(), os.path.join(path_to_save, f'model{args.current_epoch+i_epoch}.pt'))
        print(f"Saved model at {os.path.join(path_to_save, f'model{args.current_epoch+i_epoch}{args.attempt}.pt')}")

            
        torch.cuda.empty_cache()
    print('Train done')