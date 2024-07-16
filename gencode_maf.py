from Bio.AlignIO import MafIO
from Bio import AlignIO
import pandas as pd
import numpy as np
import warnings
import os 
import multiprocessing as mp
from functools import partial
import traceback
import argparse 

# Ignore pandas warnings
warnings.filterwarnings('ignore', '.*returning-a-view-versus-a-copy.*',)
warnings.filterwarnings('ignore', '.*ChainedAssignmentError.*',)
warnings.filterwarnings('ignore', '.*SettingWithCopyWarning.*',)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from Bio.pairwise2 import format_alignment
from Bio import pairwise2



def query_maf(maf_name, mafindex_name, ref_contig, bed_start, bed_end):
    # idx = MafIO.MafIndex(sqlite_file, maf_file, target_seqname)
    # print(mafindex_name, maf_name, ref_contig)
    idx = MafIO.MafIndex(mafindex_name, maf_name, ref_contig)
    #idx = MafIO.MafIndex("30way_primate_main_contigs/chr1.mafindex", "30way_primate_main_contigs/chr1.maf", "hg38.chr1")
    results = idx.search([bed_start], [bed_end])
    
    seqreqs = dict()
    species_contig_count = dict()
    
    for multiple_alignment in results:
        #AlignIO.write(multiple_alignment, "chr1_KI270708v1_random_0_205.fa", "fasta")

        for seqrec in multiple_alignment:
            # make a dicr or arr to make a row for a pandas dataframe
            entry = [[
                    seqrec.annotations["strand"], 
                    seqrec.annotations["start"], 
                    seqrec.annotations["start"]+seqrec.annotations["size"],
                    seqrec.annotations["size"], 
                    len(str(seqrec.seq)),
                    seqrec.seq.upper()
            ]]
            # Create a DataFrame
            df = pd.DataFrame(entry, columns=['strand', 'start', 'end', 'len', 'true_len', 'seq'])
    
            if seqrec.id in seqreqs.keys():
                seqreqs[seqrec.id] = pd.concat([seqreqs[seqrec.id], df])

            else:
                species_contig_count[seqrec.id] = 0
                seqreqs[seqrec.id] = df

            # species_contig_count[seqrec.id] += 1

    return seqreqs


def discard_short_contigs(seqreqs, context_len):
    new_seqreqs = dict()
    
    for species in seqreqs.keys():
        df = seqreqs[species]
        
        if df['len'].sum() >= context_len //2:
            new_seqreqs[species] = df

    return new_seqreqs


def slice_start(df, rel_start, min_row):
    
    if (df['contig'] == min_row.contig).any():
        
        # print("query_contig", df[df['contig'] == min_row.contig].index.values[0])
        
        query_contig = df[df['contig'] == min_row.contig].index.values[0]
        
        first_contig_len = df[df.index == query_contig].no_gap_seq.str.len().values[0]

        gap_size = 0
        
        # print("query_contig", query_contig, "human_contig", min_row.contig)
        # print(df)

        # print("first_contig_len", first_contig_len, "rel_start", rel_start)
        while first_contig_len <= rel_start:
            # need to check if contigs are contiguous, if so execute this code
            rel_start = rel_start - first_contig_len

            # this fails before there is a skipped contig in the beginning, you have to by row index and not by contig
            query_contig += 1 

            # if the contigs are out of order with respect to the human contig, such that the next contig in the species does not exist, exit the loop
            # or if there are no more contigs to slice:
            try:
                gap_size = df[df.index == query_contig].start.values[0] - df[df.index == query_contig-1].end.values[0]
            except IndexError:
                rel_start = 0
                print(f"No data for query_contig {query_contig, first_contig_len, rel_start}, {df}")
                break # it should just return the df as is with no slicing
            
            # need to delete the previous from the dataframe
            df = df[df.index != query_contig - 1 ]
            
            if gap_size == 0:
                first_contig_len = df[df.index  == query_contig].sliced_contig_len.values[0] #.no_gap_seq.str.len().values[0]

            elif gap_size < rel_start:
                # need to account for gap in slicing
                rel_start = rel_start - gap_size
                first_contig_len = df[df.index  == query_contig].sliced_contig_len.values[0] #.no_gap_seq.str.len().values[0]
                
            elif rel_start == 0: 
                break

            # the gap_size is the same size or larger than the rel_start
            else:
                break # this should break it out of the loop

        # this will catch when there is still remaining rel_start, but the gap_size is too large and cancels it out
        if rel_start == 0  or gap_size >= rel_start: # or rel_start == first_contig_len, if 

            # by defn this will do nothing
            pass
            df.loc[df.index == query_contig, 'sliced_contig'] = df[df.index == query_contig].sliced_contig.values
            df.loc[df.index == query_contig, 'sliced_contig_len'] = df[df.index== query_contig].sliced_contig.str.len()

        elif rel_start == first_contig_len:
            query_contig += 1
            df = df[df.index != query_contig - 1 ]

        else: 
            # slice the start contig's Seq object
            df.loc[df.index  == query_contig, 'sliced_contig'] = df[df.index  == query_contig].sliced_contig.str[rel_start:].values #.no_gap_seq.str[rel_start:].values
            df.loc[df.index  == query_contig, 'sliced_contig_len'] = df[df.index  == query_contig].sliced_contig.str.len()
        
        if df.index.isin([query_contig]).any():
            # print(query_contig, df.index.isin([query_contig]).any())
            assert df.loc[df.index  == query_contig, 'sliced_contig_len'].values[0] != 0, f"contig {query_contig} is empty!"
            
        # otherwise the next query contig is not in the dataframe so there is no point to the assertion

    return df


def slice_end(df, rel_end, max_row):
    
    gap_size = 0
    
    if (df['contig'] == max_row.contig).any():
        query_contig = df[df['contig'] == max_row.contig].index.values[0] #max_row.contig
        last_contig_len = df[df.index == query_contig].sliced_contig.str.len().values[0]

        # if rel end = 0 , then there is nothing to slice off the end
        if rel_end == 0:
            df.loc[df.index  == query_contig, 'sliced_contig'] = df[df.index == query_contig].sliced_contig.values
            df.loc[df.index  == query_contig, 'sliced_contig_len'] = df[df.index == query_contig].sliced_contig.str.len()

        else:
            
            while last_contig_len < rel_end:
                # print(query_contig, last_contig_len, rel_end) # 14 41 46
                # need to check if contigs are contiguous, if so execute this code
                rel_end = rel_end - last_contig_len
                query_contig -= 1 
                
                # we have run out of contigs, but still have an end to cutoff, should result in an empty dataframe
                if query_contig < 0:
                    break

                # gap_size = df[df.index  == query_contig+1].start.values[0] - df[df.index == query_contig].end.values[0]
                # print(query_contig, last_contig_len, rel_end, gap_size)

                try:
                    gap_size = df[df.index  == query_contig+1].start.values[0] - df[df.index == query_contig].end.values[0]
                except IndexError:
                    rel_start = 0
                    print(f"No data for query_contig {query_contig, last_contig_len, rel_end}, {df}")
                    break
            
                # need to delete the previous from the dataframe
                df = df[df.index != query_contig + 1 ]
                
                if gap_size == 0:
                    last_contig_len = df[df.index == query_contig].sliced_contig_len.values[0] #.no_gap_seq.str.len().values[0]

                elif gap_size < rel_end:

                    # raise Error
                    # need to account for gap in slicing
                    rel_end = rel_end - gap_size
                    last_contig_len = df[df.index == query_contig].sliced_contig_len.values[0] #.no_gap_seq.str.len().values[0]

                # the gap_size is the same size or larger than the rel_end
                else: 
                    print( gap_size, rel_end, "you got here")
                    # print(df)
                    break # just break out of the loop there is nothing to slice?

            #86 3 348
            #print(rel_end, gap_size, last_contig_len)
            
            if rel_end < 0:
                print("rel_end is negative", rel_end)
                raise Error
                
            elif rel_end == 0 or gap_size >= rel_end:
                df.loc[df.index == query_contig, 'sliced_contig'] = df[df.index == query_contig].sliced_contig.values
                df.loc[df.index == query_contig, 'sliced_contig_len'] = df[df.index== query_contig].sliced_contig.str.len()

            elif rel_end >= last_contig_len:
                query_contig -= 1

                # delete the contig
                df = df[df.index != query_contig  + 1]
                # this should result in an empty dataframe

            else:
                # print(query_contig, last_contig_len, rel_end)
                # slice the end contig's Seq object
                df.loc[df.index == query_contig, 'sliced_contig'] = df[df.index == query_contig].sliced_contig.str[:-rel_end].values
                df.loc[df.index == query_contig, 'sliced_contig_len'] = df[df.index == query_contig].sliced_contig.str.len()

        if df.index.isin([query_contig]).any():
            # print(query_contig, df.index.isin([query_contig]).any())
            assert df.loc[df.index  == query_contig, 'sliced_contig_len'].values[0] != 0, f"contig {query_contig} is empty!"
            
        # otherwise the next query contig is not in the dataframe so there is no point to the assertion

    return df



def slice_contigs(seqreqs, bed_start, bed_end):
    human_key = next((key for key, value in seqreqs.items() if "hg38" in key), None)
    human_df = seqreqs[human_key]
    # human_df = human_df.reset_index(drop=True)

    # get first and last contig
    min_row = human_df.loc[human_df['contig'].idxmin()]
    max_row = human_df.loc[human_df['contig'].idxmax()]

    # get relative start, relative end
    # print(min_row.start) 12288000 -> 12288672. yeah it is incomplete. 
    # for those that fail this null return and print why
    if not (min_row.start <= bed_start):
        print("contig starts after bed range starts, incomplete coverage")
        return dict()
    if not (max_row.end >= bed_end):
        print("contig ends before bed range ends, incomplete coverage")
        return dict()

    true_rel_start, true_rel_end  = bed_start - min_row.start, max_row.end - bed_end

    print("rel_start", true_rel_start, "rel_end", true_rel_end)
    

    # for each df in seqreqs, if the species possess both contigs, execute the relative slice
    for species in seqreqs.keys():
        # species = "tarSyr2.KE946745v1"
        df = seqreqs[species]
        # I have to guarantee this is in the right order if I am going to by indices. Let's assume the right order is dictated by start and end
        df = df.sort_values(['start', 'end'])
        df = df.reset_index(drop=True)
        rel_start, rel_end = true_rel_start, true_rel_end

        df['sliced_contig'] = df.no_gap_seq
        df['sliced_contig'] = df['sliced_contig'].astype(str)
        df['sliced_contig_len'] = df.no_gap_seq.str.len()

        # print(species, rel_start, rel_end)
        # print(df[['start', 'end', "contig", "no_gap_len", "sliced_contig", "sliced_contig_len"]])
        
        df = slice_start(df, rel_start, min_row)
        
        # print(df[['start', 'end', "contig", "no_gap_len", "sliced_contig", "sliced_contig_len"]])
        df = slice_end(df, rel_end, max_row)
        
        seqreqs[species] = df

    return seqreqs

# connect all on the rel_to_human, not on seq
def connect_cont_contigs(df):

    if df.empty:
        return df

    # Assuming df is your DataFrame
    df = df.sort_values(['contig']) #(['start', 'end'])

    rows = []
    current_row = df.iloc[0]
    
    current_row = {
        'strand': current_row['strand'],
        'start': current_row['start'],
        'end': current_row['end'],
        'true_len': current_row['true_len'], 
        'seq': current_row['seq'],
        'sliced_contig': current_row['sliced_contig'],
        'sliced_contig_len': current_row['sliced_contig_len'],
        'no_gap_seq': current_row['no_gap_seq'],
        'no_gap_len': current_row['no_gap_len'],
        'rel_to_human': current_row['rel_to_human'], 
        'len': current_row['len'], 
        'rel_to_human_len': current_row['rel_to_human_len'],
        'contig': str(current_row['contig'])
    }
    
    for _, row in df.iloc[1:].iterrows():
        if row['start'] == current_row['end']:
            current_row = {
                'strand': row['strand'],  # assuming 'strand' is the same for both rows
                'start': min(row['start'], current_row['start']),
                'end': max(row['end'], current_row['end']),
                'true_len': current_row['true_len'] + row['true_len'], 
                'seq': current_row['seq'] + row['seq'],
                'sliced_contig': current_row['sliced_contig'] + row['sliced_contig'],
                'sliced_contig_len': current_row['sliced_contig_len'] + row['sliced_contig_len'],
                'no_gap_seq': current_row['no_gap_seq'] + row['no_gap_seq'],
                'no_gap_len': current_row['no_gap_len'] + row['no_gap_len'],
                'rel_to_human': current_row['rel_to_human'] + row['rel_to_human'], 
                'len': max(row['end'], current_row['end']) - min(row['start'], current_row['start']), 
                'rel_to_human_len': current_row['rel_to_human_len'] + row['rel_to_human_len'],
                'contig': str(current_row['contig']) + "-" + str(row['contig'])
            }
        else:
            # if isinstance(current_row, dict):
            rows.append(current_row)
            current_row = row

    # if isinstance(current_row, dict):
    rows.append(current_row)
    merged_df = pd.DataFrame(rows)

    # we are missing the times when there is only one contig to start with
    if merged_df.empty:
        print(merged_df)
        raise Error
        return merged_df
   
    # this will eliminate enveloped rows
    # Create a mask where for each row, we check if there's any other row that satisfies the condition
    mask = merged_df.apply(lambda row: ((merged_df['start'] < row['start']) & (merged_df['end'] > row['end'])).any(), axis=1)
    
    # Invert the mask using ~, then use it to index the DataFrame. This will select only the rows where the mask is False
    merged_df = merged_df[~mask]

    # print("After Masking")
    # print(merged_df[["start", "end", "true_len", "len", "contig"]])

    
    # print(merged_df[['start', 'end', "contig", 'true_len', 'len']])
    # raise Error
    return merged_df

# connect discontinuous contigs with "-" between the contigs
def connect_discon_contigs(df):
    if df.empty:
        return df
    # Assuming df is your DataFrame
    df = df.sort_values(['start', 'end'])

    rows = []
    current_row = df.iloc[0]
    current_row = {
        'strand': current_row['strand'],
        'start': current_row['start'],
        'end': current_row['end'],
        'true_len': current_row['true_len'], 
        'seq': current_row['seq'],
        'sliced_contig': current_row['sliced_contig'],
        'sliced_contig_len': current_row['sliced_contig_len'],
        'no_gap_seq': current_row['no_gap_seq'],
        'no_gap_len': current_row['no_gap_len'],
        'rel_to_human': current_row['rel_to_human'], 
        'len': current_row['len'], 
        'rel_to_human_len': current_row['rel_to_human_len'],
        'contig': str(current_row['contig'])
    }

    for _, row in df.iloc[1:].iterrows():
        if row['start'] != current_row['end']:

            gap_size = row['start'] - current_row['end']
            # if the gap_size is too large, keep the strings separate. There is no point training on a huge gap
            if gap_size > 20:
                rows.append(current_row)
                current_row = row
                continue
            
            current_row = {
                'strand': row['strand'],  # assuming 'strand' is the same for both rows
                'start': min(row['start'], current_row['start']),
                'end': max(row['end'], current_row['end']),
                'true_len': current_row['true_len'] + row['true_len'] + gap_size, 
                'seq': current_row['seq'] + gap_size * "-" + row['seq'],
                'sliced_contig': current_row['sliced_contig'] + gap_size * "-" + row['sliced_contig'],
                'sliced_contig_len': current_row['sliced_contig_len'] + row['sliced_contig_len'] + gap_size,
                'no_gap_seq': current_row['no_gap_seq'] + gap_size * "-" + row['no_gap_seq'],
                'no_gap_len': current_row['no_gap_len'] + row['no_gap_len'] + gap_size,
                'rel_to_human': current_row['rel_to_human'] + gap_size * "-" + row['rel_to_human'], 
                'len': max(row['end'], current_row['end']) - min(row['start'], current_row['start']), 
                'rel_to_human_len': current_row['rel_to_human_len'] + row['rel_to_human_len'] + gap_size,
                'contig': str(current_row['contig']) + "x" + str(row['contig'])
            }

        # if this conditional is met, then there are still contigous rows in the dataframe
        else:
            print("Issue connecting Discontiguous Contigs")
            current_row = {
                'strand': row['strand'],  # assuming 'strand' is the same for both rows
                'start': min(row['start'], current_row['start']),
                'end': max(row['end'], current_row['end']),
                'true_len': current_row['true_len'] + row['true_len'], 
                'seq': current_row['seq'] + row['seq'],
                'sliced_contig': current_row['sliced_contig'] + row['sliced_contig'],
                'sliced_contig_len': current_row['sliced_contig_len'] + row['sliced_contig_len'],
                'no_gap_seq': current_row['no_gap_seq'] + row['no_gap_seq'],
                'no_gap_len': current_row['no_gap_len'] + row['no_gap_len'],
                'rel_to_human': current_row['rel_to_human'] + row['rel_to_human'], 
                'len': max(row['end'], current_row['end']) - min(row['start'], current_row['start']), 
                'rel_to_human_len': current_row['rel_to_human_len'] + row['rel_to_human_len'],
                'contig': str(current_row['contig']) + "-" + str(row['contig'])
            }

    rows.append(current_row)
    merged_df = pd.DataFrame(rows)
    
    return merged_df


def annotate_contigs(seqreqs):
    
    human_key = next((key for key, value in seqreqs.items() if "hg38" in key), None)
    human_df = seqreqs[human_key]
    human_df = human_df.reset_index(drop=True)
    
    # Create the map from df1
    # len_to_index_map = create_map(human_df)
    human_df["contig"] = human_df.index
    seqreqs[human_key] = human_df #apply_map(human_df , len_to_index_map)

    
    for species in seqreqs.keys():
        #species = "manLeu1.NW_012107275v1" #"cerAty1.NW_012001627v1"
        #print(species)
        if "hg38" not in species:
    
            df2 = seqreqs[species]
            df2 = df2.reset_index(drop=True) 
            df2_index = 0 
            
            for i, ref_contig in human_df.iterrows():
                # right now it will just throw an error if the index does not exist in the non_human file
                ref_contig_len = ref_contig["true_len"]

                if df2_index > len(df2.index) - 1:
                    break
                    
                # check if df2 has this contig
                elif df2.loc[df2_index,"true_len"] == ref_contig_len:
                    df2.loc[df2_index, "contig"] = int(i)

                    # update the df2 index if you saw the contig
                    df2_index += 1

                # if you did not see the contig, keep the same df2 index
                else:
                    continue
            # print("human")
            # print(human_df)
            # print(species)
            # print(df2)
            # raise Error
            df2["contig"] = df2["contig"].astype(int)

            # raise Error
            seqreqs[species] = df2
            
            # Apply the map to df2
            # seqreqs[species] = apply_map(df2 , len_to_index_map)

    return seqreqs


def remove_gaps(seqreqs):
    for species in seqreqs.keys():
        
        species_df = seqreqs[species]
        species_df['no_gap_seq'] = species_df['seq'].apply(lambda x: x.replace("-", ""))
        species_df['no_gap_len'] = species_df['no_gap_seq'].str.len()
        seqreqs[species] = species_df

    return seqreqs



def align_rel_to_human(seqreqs):
    # then do pairwise global alignment on the correct contig pairs. Given contig == 0, get contig == 0 in the non human df
    human_key = next((key for key, value in seqreqs.items() if "hg38" in key), None)
    human_df = seqreqs[human_key]

    for species in seqreqs.keys():

        species_df = seqreqs[species]

        
        # print(human_df[["start", "end", "true_len", "len", "sliced_contig_len", "contig"]])
        # print(species)
        # print(species_df[["start", "end", "true_len", "len", "sliced_contig_len", "contig"]])

        # First, set 'contig' as the index for efficient lookup
        human_lookup = human_df.set_index('contig', inplace=False)
        species_df = species_df.set_index('contig', inplace=False)
        
        # Then, iterate over the unique contigs in human_df
        for contig in human_lookup.index.unique():
            if contig in species_df.index:
                human_str = human_lookup.loc[contig, 'sliced_contig']
                non_human_str = species_df.loc[contig, 'sliced_contig']
        
                alignments = pairwise2.align.globalms(
                    human_str, 
                    non_human_str, 
                    # get 2 points for a match, -1 for a mismatch, -2 for a gap start, -0.5 for a gap extension
                    1, -1, -2, -.5,
                    one_alignment_only = True)
        
                # Get the aligned sequences
                human_return = alignments[0][0]
                non_human_return = alignments[0][1] # then take the second sequence back, use this at inference
        
                species_df.loc[contig, 'rel_to_human'] = non_human_return
                species_df.loc[contig, 'rel_to_human_len'] = len(non_human_return)
        
        seqreqs[species] = species_df.reset_index()

    return seqreqs


def merge_all_contigs(seqreqs):
    for species in seqreqs.keys():
        
        #species = "manLeu1.NW_012107275v1"
        # species = "nomLeu3.chr24"
        # print(species)
        species_df = seqreqs[species]
        # print(species_df[["start", "end", "len", "true_len","contig"]])
        # print(species_df[["start", "end", "len", "true_len", "sliced_contig_len", "no_gap_len"]])
        # print(species_df[["contig", "seq", "sliced_contig", "no_gap_seq", 'rel_to_human']])

        # we connect the contiguous ones
        species_df = connect_cont_contigs(species_df)
        # print(species_df)
        # now we connect the discontiguous ones
        species_df = connect_discon_contigs(species_df)
        # print(species_df)
        seqreqs[species] = species_df

    return seqreqs


def write_to_file(row, str_dir, species, bed_chr, bed_start, bed_end):
    # Convert the 'rel_to_human' column to a string
    rel_to_human_str = str(row['rel_to_human'])
    
    species_only, contig_only = species.split(".")
    # Write the string to a text file

    # the row.contig can be too long to write file names properly
    # break apart by existance of "x" i.e. gaps in the contigs

    # 30_way_primate_strs/hg38/chr12/0-1_chr12_102195200_102196224.txt
    continuous_contigs = str(row.contig).split("x")

     # if no "x", then split on "-". take the first and last token in this case
    cleaned_up_contigs = []
    
    for contig in continuous_contigs:
        shrunk_contig_name = contig.split("-")[0] + "-" + contig.split("-")[-1]
        cleaned_up_contigs.append(shrunk_contig_name)
        
    # recombine and write out 
    cleaned_up_contigs = "_".join(cleaned_up_contigs)

    filename = f'{str_dir}/{species_only}/{contig_only}/{cleaned_up_contigs}_{bed_chr}_{bed_start}_{bed_end}.txt'
    
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")
        
    with open(filename, 'w') as f:
        f.write(rel_to_human_str)



def full_process(row, mafindex_name, maf_name, ref_contig, str_dir):
    
    bed_chr, bed_start, bed_end = row.chrom, row.start, row.end

    print(bed_chr, bed_start, bed_end)
    
    # currently bed_chr does nothing and red_contig is used. there needs to be code to check the two.

    # bed_chr, bed_start, bed_end = "chr1", 133120, 134144
    # should return upper case only
    seqreqs = query_maf(maf_name, mafindex_name, ref_contig, bed_start, bed_end)

    # print(seqreqs)
    #print("query", seqreqs["hg38.chr21"][["true_len",]].sum())

    
    # human_df, non_human_df = seqreqs['hg38.chr3'], seqreqs['gorGor5.CYUI01015383v1']
    # print(bed_chr, bed_start, bed_end, seqreqs)
    if len(seqreqs) == 0:
        print("Maf is empty for this position!")
        return
    
    # throw out any species whose sum of len of returned contigs is less than half of the query len
    seqreqs = discard_short_contigs(seqreqs, bed_end-bed_start)
    #print(len(seqreqs.keys()))
    if len(seqreqs) == 0:
        print("All species discarded after filtering at this position!")
        return

    #print("discard short contigs", seqreqs["hg38.chr21"][["true_len",]].sum())
    # annotate contigs with human contig indices
    seqreqs = annotate_contigs(seqreqs)

    #print("annotate contigs", seqreqs["hg38.chr21"][["true_len",]].sum())
    # print(seqreqs["hg38.chr1"])
    # print(seqreqs["dasNov3.JH564546"])
    
    # is multiple contigs have the same start, get the biggest one
    for species in seqreqs.keys():
        df = seqreqs[species]
        df = df.loc[df.groupby('start')['end'].idxmax()]
    seqreqs[species] = df 

    #print("after enveloping", seqreqs["hg38.chr21"][["true_len", ]].sum())
    # remove gaps in all contigs in all species
    seqreqs = remove_gaps(seqreqs)

    #print("gaps removed", seqreqs["hg38.chr21"][["true_len", "no_gap_len", ]].sum())
    # print(seqreqs["hg38.chr1"])
    # print(seqreqs["papAnu3.chr1"])

    # slice the start and end contigs appropriately
    #seqreqs = slice_ends(seqreqs, bed_start, bed_end)
    seqreqs = slice_contigs(seqreqs, bed_start, bed_end)

    if len(seqreqs) == 0:
        print("All species discarded after slicing the contigs at this position!")
        return

    # print(seqreqs["hg38.chr1"])
    # print(seqreqs["dasNov3.JH564546"])
    # print(len(seqreqs.keys()))
    # print("contigs sliced", seqreqs["hg38.chr21"][["start", "end", "true_len", "no_gap_len", "sliced_contig_len"]])#.sum())
    # print("contigs sliced", seqreqs['macFas5.chr10'][["start", "end", "true_len", "no_gap_len", "sliced_contig_len"]])#.sum())
    
    # throw out any species whose sum of len of returned contigs is less than half of the query len
    seqreqs = discard_short_contigs(seqreqs, bed_end-bed_start)


    #print("discard short contigs again", seqreqs["hg38.chr21"][["true_len", "no_gap_len", "sliced_contig_len"]].sum())

    # print(seqreqs["hg38.chr1"])
    # print(seqreqs["papAnu3.chr1"])
    # raise Error


    # align contigs relative to human contigs
    seqreqs = align_rel_to_human(seqreqs)

    # merge all contigs together unless the gap between contigs is > 20
    seqreqs = merge_all_contigs(seqreqs)

    

    # make sure the appropriate directory is already existant, if not then make it
    os.makedirs(str_dir, exist_ok=True)
    
    for species in seqreqs.keys():
        # Apply the function to every row in the DataFrame
        species_df = seqreqs[species]
        # print(species)
        # print(species_df)

        species_only, contig_only = species.split(".")
        
        # make sure the appropriate directory is already existant, if not then make it
        os.makedirs(f"{str_dir}/{species_only}/{contig_only}", exist_ok=True)

        if not species_df.empty:
        
            species_df.apply(write_to_file, axis=1, args=(str_dir,species, bed_chr, bed_start, bed_end))
    
            
            if seqreqs[species][ "rel_to_human_len"].sum() > 750:
                print(species, seqreqs[species][ "rel_to_human_len"].sum())


# let's generalize to all chromosomes in the set

def create_mafindex(chrom):
    mafindex_name = f"30way_primate_main_contigs/{chrom}.mafindex"
    maf_name, ref_contig = f"30way_primate_main_contigs/{chrom}.maf", f"hg38.{chrom}"
    
    # if that maf index does not exist yet, make it before we start calling it in parallel
    if not os.path.exists(mafindex_name):
        print(f"creating {mafindex_name}")
        MafIO.MafIndex(mafindex_name, maf_name, ref_contig)



# need an arg parser   
def get_args():
    parser = argparse.ArgumentParser(
        description="Train the DNA LLM from frames sampled from Whole Genome Alignment"  
    )
    
    parser.add_argument("--bedfile", default = '30_way_gencode_coding_beds/chr1.bed', type =str, help = "bedfile to query maf")
    parser.add_argument("--single-threaded", action='store_true', help = "run in single threaded mode")
    parser.add_argument("--make-indices", action='store_true', help = "make maf indices")
    parser.add_argument("--str-dir", default = '30_way_gencode_coding_strs', type =str, help = "directory to store strings")
    parser.add_argument("--maf-dir", default = '30way_primate_main_contigs', type =str, help = "directory to store strings")
    return parser.parse_args() 


def apply_full_process(df):
    return df.apply(full_process_partial, axis=1)


# init from args
args = get_args()
bed_file = args.bedfile
run_single_threaded = args.single_threaded
make_indices = args.make_indices
str_dir = args.str_dir
maf_dir = args.maf_dir

chrom = bed_file.split("/")[-1].split(".")[0]
mafindex_name, maf_name, ref_contig = f"{maf_dir}/{chrom}.mafindex", f"{maf_dir}/{chrom}.maf", f"hg38.{chrom}"

# need global scope for this
full_process_partial = partial(full_process, 
                               mafindex_name=mafindex_name, 
                               maf_name=maf_name, 
                               ref_contig=ref_contig, 
                               str_dir=str_dir
                              )


# pip install pyarrow fastparquet biopython scikit-learn pyBigWig
def main():

    # init from args
    args = get_args()
    bed_file = args.bedfile
    run_single_threaded = args.single_threaded
    make_indices = args.make_indices
    str_dir = args.str_dir
    maf_dir = args.maf_dir

    # chrom = bed_file.split("/")[-1].split(".")[0]
    # mafindex_name, maf_name, ref_contig = f"{maf_dir}/{chrom}.mafindex", f"{maf_dir}/{chrom}.maf", f"hg38.{chrom}"
    
    # constructing a maf index takes a long time and is implicitly single threaded
    if make_indices:
        # Create a pool of processes
        with mp.Pool() as pool:
            pool.map(create_mafindex, chromosomes)
    
        print("All Mafindices Made!")
        raise Error

    # now we apply this for every entry in the bedfile
    col_names = ['chrom', 'start', 'end'] 
    bed_file = pd.read_csv(bed_file, sep='\t', names=col_names)

    if run_single_threaded:
        # for faster testing
        #bed_file = bed_file.iloc[326-1:]
        
        bed_file.apply(lambda row: full_process(row, mafindex_name, maf_name, ref_contig,str_dir ), axis=1)

    else:
    
        # break the bed file into smaller dataframes, then apply full process to each. 
        num_processes = mp.cpu_count()
        
        # Split the DataFrame into chunks
        chunks = np.array_split(bed_file, num_processes)
    
        # Create a pool of processes
        pool = mp.Pool(processes=num_processes)
    
        # Apply the full_process function to each chunk in parallel
        result_objects = pool.imap_unordered(apply_full_process, chunks)

        # pass through any errors that a thread encounters.
        results = []
        try:
            for result in result_objects:
                results.append(result)
        except Exception as e:
            print(f"An error of type {type(e).__name__} occurred.")
            print(f"Arguments: {e.args}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # end all processes when one process fails
            pool.terminate()
            pool.join()
            return
    
        # Close the pool
        pool.close()
        pool.join()
    
        # Combine the results
        result_df = pd.concat(results)

if __name__ == main():
    main()