import umap
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import mannwhitneyu
import os
import fnmatch
import pyBigWig
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

# make UMAPs
def make_umap(embeddings, labels, title, filename, output_dir):
    
    # Fit the UMAP model to the embeddings
    reducer = umap.UMAP(n_jobs=-1)
    labels = [int(l) for l in labels]

    embeddings_2d = reducer.fit_transform(embeddings)

    # Create a color map -- if this was extensible for WT that would be great
    colors = {
        0: f'{filename.split(".")[0].split("_")[0]}',
        1: f'{filename.split(".")[0].split("_")[-1]}'
    }
    
    # Convert labels to colors
    color_labels = [colors[i] for i in labels]
    
    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        'UMAP dimension 1': embeddings_2d[:, 0],
        'UMAP dimension 2': embeddings_2d[:, 1],
        'Color': color_labels,
        'label': labels
    })
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='UMAP dimension 1', y='UMAP dimension 2', data=df, 
                hue = 'Color',
                )
    # plt.legend(title='Labels')
    plt.title(title, fontsize=24)
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()
    
# train classifier on the embeddings or llr
def eval_classifier(feature_arr, label_arr, name, output_dir, return_yes=False):
    model = LogisticRegression(n_jobs=-1)

    scoring = {'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
               'pr_auc': make_scorer(average_precision_score, needs_proba=True)}

    scores = cross_validate(model, feature_arr, label_arr, cv=10, scoring=scoring, n_jobs=-1, return_train_score=False)

    roc_auc_scores = scores['test_roc_auc']
    pr_auc_scores = scores['test_pr_auc']

    print(f'{name} Average ROC-AUC: {roc_auc_scores.mean()}')
    print(f'{name} Average PR-AUC: {pr_auc_scores.mean()}')
    print(f'{name} positive class distribution: {len(label_arr[label_arr == 1]) / len(label_arr)}')
    print()

    # save these print statements as a csv with columns: avg ROC-AUC, PR-AUC positive class distribution
    data = {
    'Name': [name],
    'Average ROC-AUC': [roc_auc_scores.mean()],
    'Average PR-AUC': [pr_auc_scores.mean()],
    'Positive Class Distribution': [len(label_arr[label_arr == 1]) / len(label_arr)]
    }
    
    if return_yes:
        # return roc auc score if needed for immediate use
        return roc_auc_scores.mean()

    if not return_yes:
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/{name}_results.csv', index=False)


# this is a slow function just due to the for loop
def get_arrs(all_files, embed_dim, context_len, vocab_size):
    num_files = len(all_files)
    names = np.empty(num_files, dtype=object)
    WT_or_ALT_labels = np.empty(num_files, dtype=object)
    
    pathogenicity_labels =  np.zeros(num_files)
    mean_embeddings, llrs = np.zeros((num_files, embed_dim)),  np.zeros((num_files, context_len, vocab_size))
    
    
    for i in range(len(all_files)):
        input_dict = torch.load(all_files[i])
        names[i] = input_dict['name']
        llrs[i] = input_dict['llrs']
        pathogenicity_labels[i] = input_dict['pathogenicity_label']
        WT_or_ALT_labels[i] = input_dict['WT_or_ALT_labels']
        mean_embeddings[i] = input_dict['mean_embedding']#.cpu()

    return names, pathogenicity_labels, WT_or_ALT_labels, mean_embeddings, llrs


def get_true_UMAPs(WT_or_ALT_labels, mean_embeddings, pathogenicity_labels, output_dir):
    # sep WT from ALT
    WT_mask = (WT_or_ALT_labels == 'WT')
    ALT_mask = ~WT_mask
    
    WT_embeddings, WT_labels = mean_embeddings[WT_mask], pathogenicity_labels[WT_mask]
    ALT_embeddings, ALT_labels = mean_embeddings[ALT_mask], pathogenicity_labels[ALT_mask]

    # WT
    benign_mask = (WT_labels == 0)
    pathogenic_mask = ~benign_mask

    WT_benign_embeddings, WT_benign_labels = WT_embeddings[benign_mask], WT_labels[benign_mask]
    WT_pathogenic_embeddings, WT_pathogenic_labels = WT_embeddings[pathogenic_mask], WT_labels[pathogenic_mask]

    print(pathogenicity_labels.sum())
    
    # ALT
    benign_mask = (ALT_labels == 0)
    pathogenic_mask = ~benign_mask

    ALT_benign_embeddings, ALT_benign_labels = ALT_embeddings[benign_mask], ALT_labels[benign_mask]
    ALT_pathogenic_embeddings, ALT_pathogenic_labels = ALT_embeddings[pathogenic_mask], ALT_labels[pathogenic_mask]
    
    # clustering where we have WT versus Benign
    title, filename = "WT versus Benign", "WT_v_Benign.png"
    WT_Ben_embeddings, WT_Ben_labels = np.concatenate((WT_benign_embeddings, ALT_benign_embeddings)), np.concatenate((WT_benign_labels, ALT_benign_labels+1))
    make_umap(WT_Ben_embeddings, WT_Ben_labels, title, filename, output_dir)

    # clustering where we have WT versus Pathogenic
    title, filename = "WT versus Pathogenic", "WT_v_Pathogenic.png"
    WT_Path_embeddings, WT_Path_labels = np.concatenate((WT_pathogenic_embeddings, ALT_pathogenic_embeddings)), np.concatenate((WT_pathogenic_labels-1, ALT_pathogenic_labels))
    make_umap(WT_Path_embeddings, WT_Path_labels, title, filename, output_dir)
    
    # a clustering where we have WT versus ALT
    title, filename = "WT versus ALT", "WT_v_ALT.png"
    WT_ALT_embeddings, WT_ALT_labels = np.concatenate((WT_embeddings, ALT_embeddings)), np.concatenate((WT_labels, ALT_labels))
    make_umap(WT_ALT_embeddings, WT_ALT_labels, title, filename, output_dir)
    
    # clustering benign vs pathogenic ALTs
    title, filename = "Benign versus Pathogenic ALTs", "Benign_v_Pathogenic.png"
    ALT_only_embeddings, ALT_only_labels = np.concatenate((ALT_benign_embeddings, ALT_pathogenic_embeddings)), np.concatenate((ALT_benign_labels, ALT_pathogenic_labels))
    make_umap(ALT_only_embeddings, ALT_only_labels, title, filename, output_dir)    

    return [
        [WT_Ben_embeddings, WT_Ben_labels],
        [WT_Path_embeddings, WT_Path_labels],
        [WT_ALT_embeddings, WT_ALT_labels],
        [ALT_only_embeddings, ALT_only_labels],
    ]

# extract the llrs and match them up with the observed mutations 

# get violin plot of benign vs pathogenic llrs
def get_llr_violinplot(benign, pathogenic, ylabel, title, filename, output_dir):

    stat, p = mannwhitneyu(benign, pathogenic)
    print(benign.shape, pathogenic.shape)
    data = [benign, pathogenic]

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=data)
    ax.set_ylabel(ylabel) # 'LLR'
    plt.xticks([0, 1], ['Benign', 'Pathogenic'])
    plt.title(title, fontsize=24)

    # Add an asterisk if the p-value is below 0.05
    if p < 0.05:
        x1, x2 = 0, 1  # columns 'Group1' and 'Group2' (first column: 0, see plt.xticks())
        y, h, col = np.concatenate([benign, pathogenic]).max() + 1, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"* p_value = {p}", ha='center', va='bottom', color=col)

    plt.savefig(f'{output_dir}/{filename}')
    plt.close()

def map_WT_to(strings, names, pathogenicity, llrs):
    mapping = {}
    
    # context_len = 1024
    context_len = 128
    # lookup_arr = ["A", "T", "C", "G", "-"]
    lookup_arr = ["A", "T", "C", "G", "N", "-"]
    
    for i in range(len(strings)):
        string = strings[i]
        if string == "WT":
            # GPN_clinvar_1024/chr21_10649610_0_WT_1024.txt
            key = "_".join(names[i].split("/")[-1].split("_")[0:2])
            if key not in mapping:
                mapping[key] = [i]

    for i in range(len(strings)):
        string = strings[i]
        if string != "WT":# value is "to"
            key = "_".join(names[i].split("/")[-1].split("_")[0:2])
            if key in mapping:  # only add "to" strings if there's a corresponding "WT"
                mapping[key].append(i) # returns the index at which the ALT appears
    
    # at the end I want the scalar llr for the observed alt
    true_LLRs = []
    pathogenicity_subset = []
    for key, values in mapping.items():
        WT_index = values[0]

        for entry in values[1:]:
            # get the sub
            WT = strings[entry].split("_")[-3]
            logit_index = lookup_arr.index(WT)
            # print(llrs[WT_index, (context_len//2)+1, :])
            assert llrs[WT_index, (context_len//2), logit_index] == 0.0, f"{WT}, {logit_index}, {llrs[WT_index, (context_len//2), :]}"
            
            sub = strings[entry].split("_")[-1] # chr21_30215540_T_to_C
            # print(sub)
            logit_index = lookup_arr.index(sub)
            # print(logit_index)
            # print(llrs[WT_index, (context_len//2), :])
            true_LLR = llrs[WT_index, (context_len//2), logit_index]
            true_LLRs.append(true_LLR)
            pathogenicity_subset.append(pathogenicity[entry])

    true_LLRs, pathogenicity_subset = np.array(true_LLRs), np.array(pathogenicity_subset)
    
    true_LLRs = true_LLRs.reshape(true_LLRs.shape[0], 1)
    return true_LLRs, pathogenicity_subset


def fast_glob(directory, pattern):
    return [entry.path for entry in os.scandir(directory) if entry.is_file() and fnmatch.fnmatch(entry.name, pattern)]


# add the phastcons score for each snp to the df
def add_phastcons_col(bw_name, filename, contig_subset):
    bw = pyBigWig.open(bw_name)
    df = pd.read_parquet(filename) 
    df["chromosome"] = "chr"+df.chrom
    df["start"] = df.pos - 1     
    
    filled_df = []
    
    for chrom in contig_subset:
        total_len = bw.chroms()[chrom]
        df_subset = df[df.chromosome == chrom]
        df_subset["phastCons-30p"] = df_subset.apply(lambda row: (bw.values(chrom, row["start"], row["start"]+1)[0]), axis=1)
        filled_df.append( df_subset )
    
    filled_df = pd.concat(filled_df)
    return filled_df

# compute the 10-fold cv classification performance from premade scores
def eval_all_premade(filled_df, output_dir, eval_type):
    
    # this needs to change based on eval type
    if eval_type == "clinvar":
        cols_to_check_out = [
            "GPN-MSA", "CADD", "phyloP-100v", "phyloP-241m", "phastCons-100v", "ESM-1b",
            "NT", "HyenaDNA", 
            "phastCons-30p"
        ]
    else: 
        cols_to_check_out = [
            "GPN-MSA", "CADD", "phyloP-100v", "phyloP-241m", "phastCons-100v", "ESM-1b",
            # "NT", "HyenaDNA", 
            "phastCons-30p"
        ]        
    
    # get the labels as 1 or 0
    label_arr = filled_df.label.astype(int).values
    
    for col in cols_to_check_out:
        # get the features for that model
        feature_arr = filled_df[col].values
    
        # turn nans to zero if they exist
        feature_arr = np.nan_to_num(feature_arr)
    
        # separate the features into benign and pathogenic subsets for violinplots
        benign_mask = (label_arr == 0)
        benign = feature_arr[benign_mask]
        pathogenic = feature_arr[~benign_mask]
    
        # make the violinplot
        get_llr_violinplot(benign, pathogenic, col, f"Benign versus Pathogenic {col} Violinplot", 
                           f"Ben_vs_Path_{col}_violin.png", output_dir)
    
        # reshape to make sklearn happy
        feature_arr = feature_arr.reshape(-1, 1)
    
        # evaluate the classifier with 10-fold cv
        eval_classifier(feature_arr, label_arr, col, output_dir)


def make_barplot(labels, scores, eval_type, y_label, title, filename, directory):

    plt.close()
    plt.figure(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    print(labels, scores)
    plt.bar(labels, scores, color=colors)

    # plt.xlabel('Mode')
    plt.ylabel(y_label)
    plt.xticks(rotation=45) #rotation='vertical') 
    plt.title(title)
    if eval_type == "clinvar":
        plt.axhline(y=0.5, color='r', linestyle='--')
    
    plt.savefig(f"{directory}/{filename}")
    plt.close()


# search_dir = "graphs/gpn_clinvar_128"
# title = f"Autosomal {eval_type} Pathogenicity Binary Classification"

def compare_classifiers(search_dir, eval_type, title, filename):
    # get all the csv files in the chosen dir
    all_files = glob.glob(search_dir+"/*.csv")

    # read them all in as df and concatenate them
    all_dfs = [pd.read_csv(file) for file in all_files]
    all_is_one = pd.concat(all_dfs)

    # get the pathogenicity labels
    labels = all_is_one.Name.values
    
    # make a barplot from the ROC-AUC column if clinvar
    if eval_type == "clinvar":
        scores = all_is_one["Average ROC-AUC"].values
        score_name = "Average ROC-AUC"
        
    # make a barplot from the PR-AUC column if cosmic
    else:
        scores = all_is_one["Average PR-AUC"].values
        score_name = "Average PR-AUC"
    
    # sort them to make the bars look prettier
    indices = np.argsort(scores)
    labels = labels[indices]
    scores = scores[indices]

    # write a barplot in the chosen dir
    make_barplot(labels, scores, eval_type, score_name, title, filename, search_dir)


def get_embedding_results(embeddings_and_labels_sorted, eval_type, title, filename, directory):
    split_names = [
        "WT vs Benign",
        "WT vs Pathogenic",
        "WT vs ALT", 
        "Benign vs Path",
    ]

    roc_auc_list = []
    
    # train a regressor for each of these
    for i in range(len(embeddings_and_labels_sorted)):
        feature_arr = embeddings_and_labels_sorted[i][0]
        label_arr = embeddings_and_labels_sorted[i][1]
        roc_auc = eval_classifier(feature_arr, label_arr, split_names[i], directory, True)
        roc_auc_list.append(roc_auc)

    print(split_names, roc_auc_list)

    # make a barplot from these outputs
    make_barplot(split_names, roc_auc_list, eval_type, "Average ROC-AUC", title, filename, directory)


# need an arg parser   
def get_args():
    parser = argparse.ArgumentParser(
        description="Eval the DNA LLM on clinically relevant SNPs"  
    )
    # evals/GPN_clinvar_1024
    # evals/GPN_cosmic_1024
    # evals/BEND_variant_effects_expression_1024 -- this one is also annoyingly large, avoid for now
    # evals/BEND_variant_effects_disease_1024 -- this last one is huge, ignore for now
    parser.add_argument("--input-dir", default = "evals/gpn_cosmic_128", type =str, required = False, help = "directory where inputs are")
    parser.add_argument("--embed-dim", default = 128, type =int, required = False, help = "dimensionality of embeding space")
    parser.add_argument("--context-len", default = 128, type =int, required = False, help = "length of input sequence")
    parser.add_argument("--vocab-size", default = 12, type =int, required = False, help = "length of vocab for LLM")
    parser.add_argument("--bw", default = "../hg38.phastCons30way.bw", type =str, required = False, help = "bigwig for added phastcons column")
    parser.add_argument("--eval-source", default = "GPN_benchmarks/gpn_cosmic.parquet", type =str, required = False, help = "source df of SNPs for evaluation")
    
    return parser.parse_args() 

# pip install biopython scikit-learn transformers umap-learn seaborn
def main():
    # autosomes only
    contig_subset =[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7","chr8", "chr9","chr10", 
            "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17","chr18", "chr19", "chr20",
            "chr21", "chr22",
        ]

    # Parse the arguments
    args = get_args()

    # init from args
    input_dir = args.input_dir
    output_dir = f"graphs/{args.input_dir.split('/')[-1]}"
    embed_dim = args.embed_dim
    context_len = args.context_len
    vocab_size = args.vocab_size - 6 # account for all extra chars
    bw = args.bw
    eval_source = args.eval_source

    # determine the eval_type from the name of the directory
    eval_type = output_dir.split("/")[-1].split("_")[1]
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read in all pt files from dir
    all_files = fast_glob(input_dir, "*.pt") #glob.glob(input_dir)
    
    names, pathogenicity_labels, WT_or_ALT_labels, mean_embeddings, llrs = get_arrs(all_files, embed_dim, context_len, vocab_size )

    # split up the embedding and label arrs for task set
    embeddings_and_labels_sorted = get_true_UMAPs(WT_or_ALT_labels, mean_embeddings, pathogenicity_labels, output_dir)

    # make a barplot of the results
    get_embedding_results(embeddings_and_labels_sorted, eval_type, "Performance of all Embedding splits", "all_embedding_splits.png", output_dir)
    
    # get llrs
    true_llrs, true_labels = map_WT_to(WT_or_ALT_labels, names, pathogenicity_labels, llrs)

    # train a regressor to split benign vs pathogenic llrs
    eval_classifier(true_llrs, true_labels, "Model LLR", output_dir,return_yes=False)

    benign_mask = (true_labels == 0)
    benign = true_llrs[benign_mask].flatten()
    pathogenic = true_llrs[~benign_mask].flatten()
    
    # violin plot benign vs pathogenic llrs
    get_llr_violinplot(benign, pathogenic, 'LLR', "Benign versus Pathogenic LLR Violinplot", "Ben_vs_Path_violin.png", output_dir)

    # load in the val set and add the phastcons column
    filled_df = add_phastcons_col(bw, eval_source, contig_subset)
    
    # get the other violin plots
    eval_all_premade(filled_df, output_dir, eval_type)
    
    # compare all the other classifiers versus phastcons and our own llr
    compare_classifiers(output_dir, eval_type, f"Performance of all Genomics models on {eval_type}", f"all_models_{eval_type}.png")
    
    




if __name__ == "__main__":
    main()

