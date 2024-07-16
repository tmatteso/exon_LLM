import os
import torch
from training import init_tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
import torch.multiprocessing as mp
import glob
from model import ConstraintBertModel
import argparse 
import torch.distributed as dist
# no assertion for bed lookups here, make that a separate file


def read_in_string(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

def get_labels(filename):
    # f"BEND_variant_effects_disease_1024/{chromosome}_{int(position)}_{int(label)}_WT_{int(length)}.txt"
    # f"BEND_variant_effects_disease_1024/{chromosome}_{int(position)}_{int(label)}_{REF}_to_{ALT}_{int(length)}.txt"
    
    filename = filename.split("/")[-1]
    labels = filename.split("_")
    chromosome, position, pathogenicity_label = labels[:3]
    length_label = labels[-1]
    WT_or_ALT_label = labels[3: -1]

    # for ALT
    if len(WT_or_ALT_label) != 1:
        full_ALT_label = [chromosome] + [position] + WT_or_ALT_label
        WT_or_ALT_label = "_".join(full_ALT_label)

    else:
        WT_or_ALT_label = WT_or_ALT_label[0]

    return pathogenicity_label, length_label, WT_or_ALT_label

# this is for a single chromosome
class InferenceDataset(Dataset):
    # sequence_length=1024, stride=512
    def __init__(self, files):
        self.all_strings = glob.glob(files)

    def __len__(self):
        return len(self.all_strings)

    def __getitem__(self, idx):
        # this gets the seqs, need labels too
        out = dict()
        filename = self.all_strings[idx]
        
        out["seq"] = read_in_string(filename)
        pathogenicity_label, length_label, WT_or_ALT_label = get_labels(filename)
        out["pathogenicity_label"] = pathogenicity_label
        out["WT_or_ALT_label"] = WT_or_ALT_label
        out["name"] = filename
        
        return out
        
def collate_fn(batch):
    # collate list of dicts into one dict
    out_batch = dict()
    out_batch["seqs"] = [sample["seq"] for sample in batch]
    out_batch["pathogenicity_labels"] = [sample["pathogenicity_label"] for sample in batch]
    out_batch["WT_or_ALT_labels"] = [sample["WT_or_ALT_label"] for sample in batch]
    out_batch["names"] = [sample["name"] for sample in batch]
    
    return out_batch


def compare_strings(str1, str2):
    differences = []
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            differences.append((i, str1[i], str2[i]))
    for i in range(min(len(str1), len(str2)), max(len(str1), len(str2))):
        if len(str1) > len(str2):
            differences.append((i, str1[i], None))
        else:
            differences.append((i, None, str2[i]))
    return differences

# get embeddings and logits
def collect_forward_pass(model, tokenizer, batch, rank, batch_size, length, embed_dim, vocab_size):
    encoded_sequence = tokenizer.batch_encode_plus(batch["seqs"], padding=True)
    # print(batch["seqs"][0][719:725])
    # print(tokenizer.batch_decode(encoded_sequence["input_ids"])[0].replace(" ", "")[719:725])
    # print(compare_strings(batch["seqs"][0][719:725], tokenizer.batch_decode(encoded_sequence["input_ids"])[0].replace(" ", "")[719:725]))
    
    # Convert the encoded sequences to tensors
    encoded_sequence = torch.tensor(encoded_sequence["input_ids"], dtype=torch.long).to(rank)
    assert encoded_sequence.shape == (batch_size, length), f"Batch sequences are wrong shape: {encoded_sequence.shape}, should be {(batch_size, length)}, names: {batch['names']}"

    output = model.forward(encoded_sequence)
    logits, embeddings = output["logits"], output["embeddings"]
    logits = logits.permute(0, 2, 1)
    
    mean_embeddings = torch.mean(embeddings, dim=1) #.flatten()
    assert mean_embeddings.shape == (batch_size, embed_dim), f"Batch mean embeddings are wrong shape: {mean_embeddings.shape}, should be {(batch_size, embed_dim)}"

    # collect class number from the tokenizer
    assert logits.shape == (batch_size, vocab_size, length), f"Batch logits are wrong shape: {logits.shape}, should be {(batch_size, vocab_size, length)}"

    log_softmaxed_logits = torch.log_softmax(logits, dim=1).cpu().numpy()

    return encoded_sequence, mean_embeddings, log_softmaxed_logits

def compute_LLR(llr, encoded_sequence, batch_size, vocab_size, length):
    
    enc = encoded_sequence.cpu().numpy()
    # all_vals = []

    # for i in range(llr.shape[0]):
    #     vals = []
    #     for j in range(llr.shape[2]):
    #         token_log = llr[i, :, j] #.flatten()
    #         WT_log = token_log[enc[i, j]]
    #         # [6:11] -> A, C, G, T, _
    #         vals.append((token_log - WT_log)[6:11])
    #     all_vals.append(vals)    

    # all_vals = np.array(all_vals)
    
    # Expand dimensions of enc for broadcasting
    enc_exp = np.expand_dims(enc, axis=1)
    enc_exp = np.repeat(enc_exp, llr.shape[1], axis=1)

    # get the appropriate indices to perform the subtraction
    index1 = np.arange(llr.shape[0])[:, None, None]
    index2 = enc_exp
    index3 = np.arange(llr.shape[2])
    
    # Subtract WT_log from token_log
    vals = llr - llr[index1, index2, index3]
    vals = np.transpose(vals, (0, 2, 1))
    
    # Slice the array
    vals = vals[:, :, 6:12,] # was 6:11, but now have "N" in vocab

    # print(all_vals.sum())
    # print(vals.sum())
    
    assert vals.shape == (batch_size, length, vocab_size-6), f"Batch logits are wrong shape: {vals.shape}, should be {(batch_size, length,  vocab_size-6)}"

    return vals

def save_outputs(out_dir, names, mean_embeddings, all_llrs, pathogenicity_labels, WT_or_ALT_labels):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for i in range(len(names)):
        dict_obj = {
            'name': names[i], 
            'mean_embedding': mean_embeddings[i].detach().cpu().numpy(),
            'llrs': all_llrs[i],
            'pathogenicity_label': pathogenicity_labels[i],
            'WT_or_ALT_labels': WT_or_ALT_labels[i]
        }

        name = names[i].split(".")[0]
        sub_dir = name.split("/")[0]
        
        if not os.path.exists(f"{out_dir}/{sub_dir}"):
            os.makedirs(f"{out_dir}/{sub_dir}")

        
        torch.save(dict_obj, f'{out_dir}/{name}_output.pt')
        
    # save one pkl per name. pkl a dict
    


def inference(
    model, 
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, 
    data_loader: torch.utils.data.DataLoader, 
    rank: int, 
    out_dir: str,
    batch_size: int, 
    length: int,
    embed_dim: int
    ):
    vocab_size = len(tokenizer.get_vocab()) 
    
    # for batch in data_loader:
    for batch in tqdm(data_loader, desc="Processing batches"):
        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                
                encoded_sequence, mean_embeddings, log_softmaxed_logits = collect_forward_pass(model, tokenizer, batch, rank, batch_size, length, embed_dim, vocab_size)
                all_llrs = compute_LLR(log_softmaxed_logits, encoded_sequence, batch_size, vocab_size, length)
                names = batch["names"]
                
                # save each separately
                save_outputs(out_dir, names, mean_embeddings, all_llrs, batch['pathogenicity_labels'], batch['WT_or_ALT_labels'])

def check_all_parameters_bf16(model):
    return all(param.dtype == torch.bfloat16 for param in model.parameters())

def single_gpu(
    train_path: str, 
    batch_size: int, 
    length: int,
    embed_dim: int,
    model,
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, 
    out_dir: str
    ):

    rank = 0
    
    dataset1 = InferenceDataset(train_path)
    
    sampler1 = torch.utils.data.RandomSampler(dataset1)
    data_loader = torch.utils.data.DataLoader(
                                                dataset1,
                                                collate_fn=collate_fn,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                sampler=sampler1,
                                                drop_last=False,
                                                )
    # push model weights to bf16 and push to current GPU
    model = model.bfloat16().to(rank)
    
    inference(model, tokenizer, data_loader, rank, out_dir, batch_size, length, embed_dim)



def setup(rank: int, 
          world_size: int,
         ):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '443'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    
# now do it with multiple gpus
def multi_gpu(
    rank: int, 
    world_size: int, 
    train_path: str, 
    batch_size: int, 
    length: int,
    embed_dim: int,
    model,#: model.ConstraintBertModel, 
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, 
    out_dir: str, 
    ):
    
    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size # this gets the number of GPUs per node
    device_ids = list(range(rank * n, (rank + 1) * n)) # this maps the GPUs to each node
    
    dataset1 = InferenceDataset(
        train_path, 
    )
    
    sampler = DistributedSampler(
        dataset1, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False, 
        drop_last=False
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset1, 
        collate_fn=collate_fn,
        batch_size=batch_size, # needs to be called
        pin_memory=True, 
        #num_workers=0, 
        drop_last=False, 
        shuffle=False, 
        sampler=sampler
    )
    
    # initialize PyTorch distributed using environment variables (you could also pass in arguments manually)
    setup(rank, world_size)

    # push model weights to bf16 and push to current GPU
    model = model.bfloat16().to(rank)

    if not check_all_parameters_bf16(model):
        raise ValueError("Not all model parameters are in bf16!")

    print(device_ids)
    
    # execute the DDP wrap
    model = DDP(model.to(device_ids[0]), device_ids=device_ids)

    inference(model, tokenizer, data_loader, rank, out_dir, batch_size, length, embed_dim)
    dist.destroy_process_group()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# need an arg parser   
def get_args():
    parser = argparse.ArgumentParser(
        description="Eval the DNA LLM on clinically relevant SNPs"  
    )
    # BEND_variant_effects_disease_1024
    
    parser.add_argument("--eval-dir", default = "gpn_cosmic_128/*", type =str, required = False, help = "directory where DNA frames are")
    parser.add_argument("--DDP", action='store_false', help = "Use DDP to train with multiple GPUs")
    parser.add_argument("--checkpoint-name", default="gencode_30way_primate_epoch_0.pth", type =str, help = "define checkpoint name")
    parser.add_argument("--batch-size", default=1, type =int, help = "how many sequences to process in a batch") 
    parser.add_argument("--context-length", default=128, type =int, help = "length of frames")  
    parser.add_argument("--embed-dim", default=128, type =int, help = "embedding dimension of model") 
    parser.add_argument("--out-dir", default = "evals", type =str, required = False, help = "directory where outputs will be stored")
    
    return parser.parse_args() 



# pip install biopython scikit-learn transformers
def main():

    # Parse the arguments
    args = get_args()

    # init from args
    embed_dim = args.embed_dim
    length = args.context_length
    out_dir = args.out_dir
    multi_GPU = args.DDP
    batch_size = args.batch_size
    WORLD_SIZE = torch.cuda.device_count()
    eval_path = args.eval_dir # "30_way_primate_strs"
    checkpoint_name = args.checkpoint_name
    
    # init the tokenizer
    tokenizer = init_tokenizer()
    # assert tokenizer.get_vocab() == {'<CLS>': 5, '<UNK>': 2, '<MASK>': 4, '<BOS>': 0, '-': 10, '<EOS>': 1, 'T': 7, '<PAD>': 3, 'C': 8, 'G': 9, 'A': 6}, f"tokenizer has unexpected vocabulary: {tokenizer.get_vocab()} != set('<CLS>': 5, '<UNK>': 2, '<MASK>': 4, '<BOS>': 0, '-': 10, '<EOS>': 1, 'T': 7, '<PAD>': 3, 'C': 8, 'G': 9, 'A': 6)"
    
    assert tokenizer.get_vocab() == {'A': 6, '-': 11, '<CLS>': 5, '<UNK>': 2, '<EOS>': 1, '<BOS>': 0, 'T': 7, '<MASK>': 4, 'G': 9, 'C': 8, '<PAD>': 3, 'N': 10}, f"tokenizer has unexpected vocabulary: {tokenizer.get_vocab()} != set('A': 6, '-': 11, '<CLS>': 5, '<UNK>': 2, '<EOS>': 1, '<BOS>': 0, 'T': 7, '<MASK>': 4, 'G': 9, 'C': 8, '<PAD>': 3, 'N': 10)"
    
    # init the model
    model = ConstraintBertModel(tokenizer)


    
    # load the state dict from the checkpoint location
    checkpoint = torch.load(checkpoint_name) 

    # discrepancy between how the model is saved and the model file
    new_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
    
    model.load_state_dict(new_state_dict) 
    
    # count param number
    num_parameters = count_parameters(model)
    print('The model has {:.2e} parameters.'.format(num_parameters))


    if multi_GPU:

        mp.spawn(multi_gpu,
                args = (
                    WORLD_SIZE, 
                    eval_path,  
                    batch_size,
                    length,
                    embed_dim,
                    model, 
                    tokenizer,
                    out_dir
                ),
                nprocs = WORLD_SIZE,
                join = True
                )
    else:
        single_gpu(
            eval_path,  
            batch_size,
            length,
            embed_dim,
            model, 
            tokenizer,
            out_dir
        )


if __name__ == "__main__":
    main()
