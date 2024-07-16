from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse 
from model import ConstraintBertModel
from dist_utils import single_GPU_main, ddp_main
import glob

def fast_glob(directory, pattern):
    return [entry.path for entry in os.scandir(directory) if entry.is_file() and fnmatch.fnmatch(entry.name, pattern)]

# this is for a single chromosome
class TextFileDataset(Dataset):
    def __init__(self, 
                 directory: str, 
                 read_from_arr: bool,
                ):

        if not read_from_arr:
            self.file_names = np.array(glob.glob(f'{directory}/*/*/*.txt'))
            # self.file_names = np.array(fast_glob(directory, "/*/*/*.txt"))
            print("before filtering", f'{directory}/*/*/*.txt', len(self.file_names ))

            # filter the strings with respect to size
            sizes = np.array([os.path.getsize(filename) for filename in self.file_names])

            self.file_names[(sizes >= 88) & (sizes <= 128)]
            sizes = sizes[(sizes >= 88) & (sizes <= 128)]
            
            # Create histogram
            plt.hist(sizes, bins=30)
            
            # Save histogram as an image file
            plt.savefig('size_hist.png')
            
            # sort the indices such that similar sized files are close together
            indices = np.argsort(-sizes) # will put longest sequences first
            self.file_names = self.file_names[indices]
            print("after filtering", len(self.file_names ))

            np.save("file_name_arr.npy", self.file_names)

        else:
            self.file_names = np.load("file_name_arr.npy")
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with open(self.file_names[idx], 'r') as f:
            content = f.read()

        # name = self.file_names[idx]
        # return name, content
        return content

def collate_fn(batch):
    # names = [item[0] for item in batch] 
    # batch = [item[1] for item in batch] 
    # return names, batch 
    return [text for text in batch] 


def init_tokenizer():
    # currently we are missing Ns in our alphabet. This must be noted as it forces us to remove some frames
    
    # Define the vocabulary
    vocab = ["A", "T", "C", "G", "N", "-"]
    # vocab = ["A", "T", "C", "G", "-"]
    
    # Define the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(BPE()),
        tokenizer_file=None,
        vocab_files=None,
        bos_token="<BOS>", #"<s>",
        eos_token='<EOS>', #"</s>",
        unk_token="<UNK>", # '<CLS>'
        pad_token='<PAD>', #"<pad>",
        mask_token='<MASK>', #"<mask>",
        additional_special_tokens=['<CLS>'],
    )

    tokenizer.add_tokens(vocab)
    return tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# need an arg parser   
def get_args():
    parser = argparse.ArgumentParser(
        description="Train the DNA LLM from frames sampled from Whole Genome Alignment"  
    )
    
    parser.add_argument("--train-dir", default = "30_way_gencode_coding_strs", type =str, required = False, help = "directory where DNA frames are")
    parser.add_argument("--epoch-num", default = 1, type = int, help = "Number of epochs")
    parser.add_argument("--use-wandb", action='store_true', help = "Use Wandb to track training")
    parser.add_argument("--DDP", action='store_false', help = "Use DDP to train with multiple GPUs")
    parser.add_argument("--read-from-arr", action='store_false', help = "Use presaved file list to avoid globbing again")
    parser.add_argument("--checkpoint-name", default="gencode_30way_primate.pth", type =str, help = "define checkpoint name")
    
    return parser.parse_args() 



# pip install transformers 

def main():

    # Parse the arguments
    args = get_args()
    torch.autograd.set_detect_anomaly(True)

    # init from args
    epoch_num = args.epoch_num
    use_wandb = args.use_wandb
    multi_GPU = args.DDP
    WORLD_SIZE = torch.cuda.device_count()
    train_path = args.train_dir # "30_way_primate_strs"
    read_from_arr = args.read_from_arr
    checkpoint_name = args.checkpoint_name
    
    # init the tokenizer
    tokenizer = init_tokenizer()
    
    # init the model
    model = ConstraintBertModel(tokenizer)

    # count param number
    num_parameters = count_parameters(model)
    print('The model has {:.2e} parameters.'.format(num_parameters))

    # init the optim
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)# lr=5e-5)
    
    # may want a scheduler later
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # ignore the padding token in loss calculation
    PAD_IDX = tokenizer.encode('<PAD>')[0]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if multi_GPU:
        mp.spawn(ddp_main,
                args = (
                    WORLD_SIZE, 
                    train_path, 
                    read_from_arr,
                    epoch_num, 
                    use_wandb,
                    criterion, 
                    model, 
                    optimizer, 
                    tokenizer,
                    checkpoint_name,
                ),
                nprocs = WORLD_SIZE,
                join = True
                )
        if use_wandb:
            wandb.finish()
    else:
        if use_wandb:
            wandb.init(project="ConstraintBERT")

        single_GPU_main(
            train_path, 
            read_from_arr,
            epoch_num, 
            use_wandb,
            criterion, 
            model, 
            optimizer, 
            tokenizer,
            checkpoint_name,
        )

        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()