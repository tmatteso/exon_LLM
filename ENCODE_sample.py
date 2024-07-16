
import pyBigWig
import numpy as np
import os
import multiprocessing as mp
import pandas as pd



# vast majority will be 250 bases or less

def read_in_gtf(gtf_file):

    # Define the column names for the GTF file
    col_names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes']
    
    # Load the GTF file
    df = pd.read_csv(gtf_file, sep='\t', comment='#', names=col_names)

    # get only the coding regions
    df = df[(df['feature'] == 'exon')]
    
    # Calculate the difference between 'end' and 'start'
    df['length'] = df['end'] - df['start'] + 1 # these gtf entries are a bit weird

    # filter for frames 512 in length or less
    df = df[df['length'] <= 512]

    # get only chrom and pos cols
    df = df[["seqname", "start", "end"]]

    # starts are off by 1! -> bed files are zero indexed
    df["start"] = df.start - 1

    return df

def write_out_chr_beds(df, dirname):
    
    os.makedirs(dirname, exist_ok=True)

    acceptable_contigs = [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7","chr8", "chr9","chr10", 
        "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17","chr18", "chr19", "chr20",
        "chr21", "chr22",
    ]
    
    for i in range(len(acceptable_contigs)):
        chrom_pos = df[df.seqname == acceptable_contigs[i]][['start', 'end']]
        # remove duplicates
        chrom_pos = chrom_pos.drop_duplicates(subset=['start', 'end'], keep='first')
        
        
        # Add the contig name to the DataFrame
        chrom_pos.insert(0, 'chrom', acceptable_contigs[i])
        
        # Write the DataFrame to a BED file
        bed_name = f'{dirname}/{acceptable_contigs[i]}.bed'
        chrom_pos.to_csv(bed_name, sep='\t', header=False, index=False)


# pip install pyBigWig
def main():

    # just get the beds in a convertible format for polished_maf to use
    gtf_file = 'gencode.v41.annotation.gtf'
    df = read_in_gtf(gtf_file)

    dirname = "30_way_gencode_coding_beds"
    write_out_chr_beds(df, dirname)
    # get the frames
    
    
    # train on the frames

    # go to bed

if __name__ == main():
    main()