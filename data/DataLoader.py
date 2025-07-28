import os
import fm
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv

import torch
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_and_stack(list_of_ndarrays):
    """
    Convert a list of [L_i, D] numpy arrays (variable‐length sequences)
    into a single tensor of shape [B, L_max, D] with zero-padding.
    """
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in list_of_ndarrays]
    return pad_sequence(tensors, batch_first=True)          # [B, L_max, D]




#   Code for Embeding generation and Integration
def check_model_cache(CACHE_PATH) -> str:
    """
    Check if RNA-FM model is available in cache.

    Returns:
     Path of the model stored in cache (so that we dont need to download everytime)
    """

    cache_path = Path.home() / CACHE_PATH
    
    if cache_path.exists():
        return str(cache_path)
    else:
        return None
   



def load_rna_fm_fast(model_path=None):
    torch.serialization.add_safe_globals([argparse.Namespace])

    if model_path and os.path.exists(model_path):
        model, alphabet = fm.pretrained.load_model_and_alphabet_local(
            model_path,
            theme="rna"
        )
    else:
        print("Model Downloading....")
        model, alphabet = fm.pretrained.rna_fm_t12()

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter




def generate_embeddings(seq_list, model, alphabet, batch_converter):
    modified_seq = []
    for i, seq in enumerate(seq_list, 1):
        modified_seq.append( (str(i), seq ))

    model.eval()
    batch_labels, batch_strs, batch_tokens = batch_converter(modified_seq)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    token_embeddings = results["representations"][12]
    token_embeddings_np = token_embeddings.cpu().numpy()
    num_seq = token_embeddings.shape[0]

    final_embeddings = []
    for i in tqdm(range(num_seq)):
        final_embeddings.append(token_embeddings_np[i])

    return final_embeddings




def load_RNAFM_and_data(data, CACHE_PATH, bio_features_return=False):    

    # data, UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN = input_to_inference(data_pre, mRNA_seq)

    cache_model_path = check_model_cache(CACHE_PATH)
    model, alphabet, batch_converter = load_rna_fm_fast(cache_model_path)
    
    model.eval()
    # sense_seq = data['Sense'].to_list()
    siRNA_seq, mRNA_seq = data['Antisense'].to_list(), data['mrna'].to_list()
    

    siRNA_embeddings = generate_embeddings(siRNA_seq, model, alphabet, batch_converter)
    mRNA_embeddings = generate_embeddings(mRNA_seq, model, alphabet, batch_converter)
    bio_features_df = data.drop(['Antisense', 'mrna', 'label', 'y'], axis=1)                        

    if bio_features_return:
        return siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df
        
    return siRNA_seq, siRNA_embeddings, mRNA_embeddings # UP_LEN, SI_LEN, DOWN_LEN, MRNA_LEN




def transform_data(data_path):
    hu_path = data_path + 'Hu.csv'
    mix_path = data_path + 'Mix.csv'
    taka_path = data_path + 'Taka.csv'

    hu_df = pd.read_csv(hu_path)
    mix_df = pd.read_csv(mix_path)
    taka_df = pd.read_csv(taka_path)

    combined_df = pd.concat([hu_df, mix_df, taka_df], axis=0)
    combined_df = combined_df.sample(frac=1)

    col_names = [
        "delta_G_end",                
        "delta_G1",                   
        "delta_H1",                   
        "U1",                         
        "G1",                         
        "delta_H_all",               
        "U_all",                     
        "U1_neighbors",              
        "G_all",                     
        "GC1",                       
        "GG1_neighbors",            
        "GC_all",                   
        "delta_G2",                 
        "UA_all",                   
        "U2",                       
        "C1",                       
        "C_all",                    
        "delta_G18",                
        "GC1_neighbors",           
        "GC_total",                
        "GC1_gc",                  
        "delta_G13",               
        "UA_all_2",                
        "A19"                 
    ]


    clean = (
        combined_df["td"]              
        .str.strip()                 
        .str.replace(" ", "", regex=False)
    )

    split_df = clean.str.split(",", expand=True).astype(float)
    split_df.columns = col_names

    combined_df.rename(columns={'siRNA':'Antisense', 'mRNA':'mrna'}, inplace=True)
    combined_df = pd.concat([combined_df.drop(columns="td"), split_df], axis=1)


    return combined_df




def preprocess_data(base_path, CACHE_PATH, bio_features_return=False):


    combined_df = transform_data(base_path)

    if bio_features_return:
        siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df = load_RNAFM_and_data(combined_df, CACHE_PATH, bio_features_return)    
        return siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df

    siRNA_seq, siRNA_embeddings, mRNA_embeddings = load_RNAFM_and_data(combined_df, CACHE_PATH, bio_features_return)
    return siRNA_seq, siRNA_embeddings, mRNA_embeddings




if __name__ == '__main__':

    dotenv_path = Path(__file__).parent.parent / "Config" / ".env"
    load_dotenv(dotenv_path)

    base_path = os.getenv("DATA_PATH")
    CACHE_PATH = os.getenv("CACHE_PATH")

    combined_df = transform_data(base_path)


    #   Can use the biological features already present in the dataset
    bio_features_return = False
    if bio_features_return:
        siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df = load_RNAFM_and_data(combined_df, CACHE_PATH, bio_features_return)    

    siRNA_seq, siRNA_embeddings, mRNA_embeddings = load_RNAFM_and_data(combined_df, CACHE_PATH, bio_features_return)