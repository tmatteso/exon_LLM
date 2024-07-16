# exon_LLM

## Download the encode exon csv and the 30-way whole genome alignment

`wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz`

Collect main autosomal contigs from http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz30way/maf/

Collect the clinvar and cosmic benchmarks from GPN-MSA
https://huggingface.co/datasets/songlab/clinvar
https://huggingface.co/datasets/songlab/cosmic

## Get bed coords from encode exons

`ENCODE_frames.py`

## Sample frames based on these coords from the WGA

`polished_maf.py`
You will need to construct the mafindices before you query frames at scale. Single threaded mode exists for debugging, but multithreaded mode is recommended for actual use. Depending on your hardware, it should take a couple minutes to get all the frames from a chromosome. There are intermediate print statements in the code currently, but these are for debugging purposes.

## Train the model from the sampled frames

`training.py` (model.py dist_utils.py)

The training.py script runs the training pipeline, whereas the model.py code houses the model architecture itself and its’ forward pass. The dist_utils.py code contains the training loop for single and multi GPU (DDP, not FSDP) modes. A model state dict should be saved after each epoch of training.

## Isolate the evaluation frames

collect_eval_frames.py

Collects context_len sized centered frames from the original eval SNP locations. Need to run for each evaluation set separately.

## Get the embeddings and logit scores for the evaluation frames

`eval.py`

This script will create a “.pt” file for each eval_frame, complete with name, pathogenicity label, embedding from the model, and logits from the model.

## Make the graphs

`graphs.py`

This script will parse the logits based on the observed ALT variants, and make graphs describing model perfomrance
