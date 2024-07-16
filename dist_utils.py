import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import transformers
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
import os

# NEED TO PASS ATTENTION MASK THAT KNOWS WHERE THE PADS ARE!
def epoch(
    rank: int, 
    train_loader: torch.utils.data.dataloader.DataLoader, 
    epoch_num: int, 
    use_wandb: bool, 
    criterion: torch.nn.modules.loss.CrossEntropyLoss, 
    model, #: model.ConstraintBertModel, 
    optim: torch.optim.Adam, 
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    ):

    print("number of batches in train loader", len(train_loader))

    loss, total_loss = 0, 0
    
    pbar = tqdm(train_loader, desc=f"Processing rank {rank}, loss: {loss}")
    
    for batch_i, data in enumerate(pbar):
        with autocast(dtype=torch.bfloat16):
        # with autocast(dtype=torch.float32):
            # names, data = data # tuple unpack the tuple of lists
            encoded_sequence = tokenizer.batch_encode_plus(data, padding=True)
    
            # Convert the encoded sequences to tensors
            encoded_sequence = torch.tensor(encoded_sequence["input_ids"], dtype=torch.long).to(rank)
            # print(tokenizer.batch_decode(encoded_sequence))

            # compute the loss
            output = model.forward(encoded_sequence)
            logits, embeddings = output["logits"], output["embeddings"]
            logits = logits.permute(0, 2, 1)
            loss = criterion(logits, encoded_sequence)

            # if nan in loss, report the sequence that caused it
            if torch.isnan(loss).any():
                print( tokenizer.batch_decode((encoded_sequence)))
                raise Error

        loss.backward()
        pbar.set_description(f"Processing rank {rank}, loss: {loss:.4f}")
        
        total_loss += loss.item()
        optim.step()
        optim.zero_grad(set_to_none=True)

        # catch gradients across all gpus for wandb tracking (this is more applicable in fsdp)
        if use_wandb:
            model_names = []
            grad_dict = dict()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:#  param.grad is not None:
                    # Gather the gradients from all GPUs
                    grad = param.grad #.sum()
  
                    zero_threshold = 1e-6 
                    grad_near_zero_count = torch.sum(grad.detach().cpu().abs() < zero_threshold).item()
                    grad_total_count = grad.detach().cpu().numel()
                    grad_near_zero_fraction = grad_near_zero_count / grad_total_count
                    
                    # grad_mean = torch.mean(grad.detach().cpu()).item()
                    # grad_std = torch.std(grad.detach().cpu()).item() if grad_total_count > 1 else 0
                    
                    one_grad = {
                        # f"gradients/{name}_mean": grad_mean,
                        # f"gradients/{name}_std": grad_std,
                        f"gradients/{name}_near_zero_fraction": grad_near_zero_fraction,
                        }
                    grad_dict.update(one_grad)
                    
                    #print(rank, type(grad_dict))
                    model_names.append(name)
                    
            # log the gradients in wandb
            if len(model_names) != 0:
                # track the activations via stats and histogram over time
                zero_threshold = 1e-6 
                matmask_near_zero_count = torch.sum(output["matmask"].abs() < zero_threshold).item()
                matmask_total_count = output["matmask"].numel()
                matmask_near_zero_fraction = matmask_near_zero_count / matmask_total_count
    
                wandb.log({
                    "epoch": epoch_num,
                    "train_loss": loss.item(),
                    "epoch_progress": accumulation_counter / len(train_loader),
                    "sample_seq_len": len(encoded_sequence),
                    "matmask_activation_mean": torch.mean(output["matmask"].detach().to(torch.float32).cpu()).item(),
                    "matmask_activation_std": torch.std(output["matmask"].detach().to(torch.float32).cpu()).item(),
                    "matmask_activation_near_zero_fraction": matmask_near_zero_fraction,
                    # f"gradients/{name}_{stat}": grad_dict[f"gradients/{name}_{stat}"] for name in model_names for stat in ["near_zero_fraction"]
                })

def check_all_parameters_bf16(model):
    return all(param.dtype == torch.bfloat16 for param in model.parameters())


# this one is for simple, single GPU training
def single_GPU_main(
    train_path: str, 
    read_from_arr: bool, 
    epoch_num: int, 
    use_wandb: bool, 
    criterion: torch.nn.modules.loss.CrossEntropyLoss, 
    model, #: model.ConstraintBertModel, 
    optim: torch.optim.Adam, 
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    checkpoint_name: str
    ):
    
    rank = 0 # cuda device zero
    world_size = 1
    # push model weights to bf16 and push to current GPU
    model = model.bfloat16().to(rank)

    if not check_all_parameters_bf16(model):
        raise ValueError("Not all model parameters are in bf16!")

    from training import TextFileDataset, collate_fn
    dataset1 = TextFileDataset(train_path, read_from_arr)
    
    sampler1 = torch.utils.data.RandomSampler(dataset1)
    train_loader = torch.utils.data.DataLoader(
                                                dataset1,
                                                collate_fn=collate_fn,
                                                batch_size=4,
                                                pin_memory=True,
                                                sampler=sampler1,
                                                drop_last=False,
                                                )
    
    for e in range(epoch_num):
        #for batch_i, (data) in enumerate(train_loader):
        epoch(
            rank,  
            train_loader, 
            epoch_num, 
            use_wandb, 
            criterion, 
            model, #: model.ConstraintBertModel, 
            optim, 
            tokenizer,
        )
        

        # save a model checkpoint
        name, extension = checkpoint_name.split(".")
        full_name = f"{name}_epoch_{i}.pth"
        torch.save(model.state_dict(), full_name)
        print(f"model saved at {full_name}")



def setup(rank: int, 
          world_size: int,
         ):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '443'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



def ddp_main(
    rank: int, 
    world_size: int, 
    train_path: str, 
    read_from_arr: bool, 
    epoch_num: int, 
    use_wandb: bool, 
    criterion: torch.nn.modules.loss.CrossEntropyLoss, 
    model, #: model.ConstraintBertModel, 
    optim: torch.optim.Adam, 
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    checkpoint_name: str
    ):
    
    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size # this gets the number of GPUs per node
    device_ids = list(range(rank * n, (rank + 1) * n)) # this maps the GPUs to each node


    # set up the training dataset and sampler
    from training import TextFileDataset, collate_fn
    
    dataset1 = TextFileDataset(
        train_path, 
        read_from_arr
    )
    
    sampler = DistributedSampler(
        dataset1, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False, 
        drop_last=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset1, 
        collate_fn=collate_fn,
        batch_size=4, 
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

    for i in range(epoch_num):
        if rank == 0:
            print(f"Epoch {i}")
        # if we are using DistributedSampler, we have to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch_num)  
        
        epoch(
            rank, 
            train_loader, 
            epoch_num, 
            use_wandb, 
            criterion, 
            model, #: model.ConstraintBertModel, 
            optim, 
            tokenizer,
        )

        if rank == 0:
            # save a model checkpoint
            name, extension = checkpoint_name.split(".")
            full_name = f"{name}_epoch_{i}.pth"
            torch.save(model.state_dict(), full_name)
            print(f"model saved at {full_name}")
        
    dist.destroy_process_group()
