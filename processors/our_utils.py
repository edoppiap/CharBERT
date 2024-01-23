import pandas as pd
import os
def clean_source(source: str, folder_level: int, root = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/mock_data/mock_code/'):
    serie_diz = {}
    candidate = source.replace(root, '')
    levels_name = candidate.split('/')

    for level in range(folder_level-1):
        serie_diz['level_'+str(level)] = levels_name[level]
    
    serie_diz['level_'+str(folder_level-1)] = '/'.join(levels_name[folder_level-1:])

    return serie_diz

#creazione dati per visualizzare su projector tensorflow
def save_data_metadata_tsv(db, emb_file_name: str, meta_file_name: str, folder_level: int, out_root_path = "/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/embeddings"):
    res = db.get( include = ['embeddings', 'metadatas'])
    
    #for the projector you need a file with only numerical data
    if not emb_file_name.endswith('.tsv'):
        emb_file_name = emb_file_name + '.tsv'
    
    #and another file with metadata
    if not meta_file_name.endswith('.tsv'):
        meta_file_name = meta_file_name + '.tsv'

    df_emb = pd.DataFrame()
    for emb in res['embeddings']:
        df_emb = df_emb.append(pd.Series(emb), ignore_index=True)
    #df_emb.to_csv(os.path.join(out_root_path, emb_file_name), sep='\t', header=False, index=False)
    
    if folder_level <= 0:
        print("Folder level must be positive and bigger that 0")
        return None

    cols = ['level_'+str(i) for i in range(folder_level)]
    df_emb_meta = pd.DataFrame(columns=cols)
    for meta_dict in res['metadatas']:
        new_row = pd.Series(clean_source(meta_dict['source'], folder_level))
        pd.concat([df_emb_meta, new_row.to_frame().T], ignore_index = True)
    
    if folder_level == 1:
        df_emb_meta.to_csv(os.path.join(out_root_path, meta_file_name), sep='\t', header=False, index=False)
    else:
        df_emb_meta.to_csv(os.path.join(out_root_path, meta_file_name), sep='\t', header=True, index=False)