import argparse
import pandas as pd
import os
from Bio import SeqIO



def get_sequence(row, seq_record, length, name):
    #print(row)
    # assert all are snps

    if 'end' in row:
        assert row.start == row.end
    
    chromosome, position = row.chromosome, row.start
    label, REF, ALT = row.label, row.ref, row.alt

    if isinstance(label, bool):
        label = int(label)

    # slice the reference genome
    start = max(0, position - length//2)
    end = start + length
    #return seq_record.seq[start:end]
    str_to_write = str(seq_record.seq[start:end]).upper()

    # make the directory of interest
    dirname = f'{name}_{length}'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # save WT
    assert len(str_to_write) == length
    assert REF == str_to_write[(length//2)], f"{REF} != {str_to_write[(length//2)-1: (length//2)+2]}"
    
    name_info = f"{chromosome}_{int(position)}_{int(label)}_WT_{int(length)}.txt"
    save_string(str_to_write, dirname, name_info)
    
    # save ALT
    ALT_str = str_to_write[:(length//2)] + ALT + str_to_write[(length//2) + 1:]
    assert len(ALT_str) == length
    assert ALT == ALT_str[(length//2)], f"ALT allele was not substitued correctly: {ALT} != {ALT_str[(length//2)-1: (length//2)+2]}"

    name_info = f"{chromosome}_{int(position)}_{int(label)}_{REF}_to_{ALT}_{int(length)}.txt"
    save_string(str_to_write, dirname, name_info)
            
    

def save_string(str_to_write, dirname, name_info):
    with open(f'{dirname}/{name_info}', 'w') as file:
        file.write(str(str_to_write))


def get_args():
    parser = argparse.ArgumentParser(
        description="Eval the DNA LLM on clinically relevant SNPs"  
    )
    # "BEND/data/variant_effects/variant_effects_disease.bed",
    # "BEND/data/variant_effects/variant_effects_expression.bed"
    # "GPN_benchmarks/gpn_clinvar.parquet", # True/False -> 54% True == clinvar pathogenic missense, False == gnomad common missense
    # "GPN_benchmarks/gpn_cosmic.parquet", # True/False -> 99% false. True == COSMIC missense, False == gnomad common missense
    # "GPN_benchmarks/gpn_gnomad.parquet", # True/False -> 97.7% true. True == gnomad rare, False == gnomad common
    # "GPN_benchmarks/gpn_omim.parquet",# True/False -> 100% false. True == OMIM pathogenic regulatory, False == gnomad common regulatory
    
    # evals/BEND_variant_effects_expression_1024 -- this one is also annoyingly large, avoid for now
    # evals/BEND_variant_effects_disease_1024 -- this last one is huge, ignore for now
    parser.add_argument("--input-file", default = "GPN_benchmarks/gpn_cosmic.parquet", type =str, required = False, help = "input file")
    parser.add_argument("--embed-dim", default = 128, type =int, required = False, help = "dimensionality of embedding space")
    parser.add_argument("--context-len", default = 128, type =int, required = False, help = "length of input sequence")
    parser.add_argument("--vocab-size", default = 12, type =int, required = False, help = "length of vocab for LLM")
    
    return parser.parse_args() 


# pip install biopython scikit-learn transformers pyarrow fastparquet 
def main():

    # Parse the arguments
    args = get_args()

    # init from args
    filename = args.input_file
    embed_dim = args.embed_dim
    context_len = args.context_len
    vocab_size = args.vocab_size - 6 # account for all extra chars
    output_dir = f"{args.input_file.split('/')[-1].split('.')[0]}"
    parquet = (filename.split(".")[-1] == "parquet")

    if parquet:
        df = pd.read_parquet(filename) 
        df["chromosome"] = "chr"+df.chrom
        df["start"] = df.pos - 1 

    else:
        df = pd.read_csv(filename, sep="\t")


    # contig_subset = list(df.chromosome.unique())  
    
    # autosomes only:
    contig_subset =[
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7","chr8", "chr9","chr10", 
        "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17","chr18", "chr19", "chr20",
        "chr21", "chr22",
    ]
    print(df.head())
    print(contig_subset)

    fasta_file = "data/hg38.fa"
    parsed_fasta = SeqIO.parse(fasta_file, "fasta")

    for seq_record in parsed_fasta:
        chrom = seq_record.id
        if chrom in contig_subset:
            
            df_subset = df[df.chromosome == chrom]
            df_subset.apply(lambda row: get_sequence(row, seq_record, context_len, output_dir), axis=1)





if __name__ == "__main__":
    main()