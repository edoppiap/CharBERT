import os
import pandas as pd
import numpy as np

def w_avg_dfs(result_jsons_path: str = 'Ir_results/'):
    json_files = [pos_json for pos_json in os.listdir(result_jsons_path) if pos_json.endswith('.json')]
    dataframes = {}
    top_at_arrays = []
    weights = []
    for idx, js in enumerate(json_files):
        with open(os.path.join(result_jsons_path, js)) as json_file:
            tmp_df = pd.read_json(json_file)
            dataframes[idx] = tmp_df
            
        weights.append(tmp_df.num_queries.iloc[0])
        tmp_top_at_array = np.vstack(tmp_df.top_at.apply(lambda x: np.array(x)).to_list())
        top_at_arrays.append(tmp_top_at_array)
    
    whole_top_at_np = np.stack(top_at_arrays)
    
    weighted_mean_np = np.average(whole_top_at_np, axis=0, weights=np.array(weights))
    
    check_feat = ['embedder', 'chunk_size', 'search_type','code_query']
    flag = True
    for i in range(len(dataframes)-1):
        df_A = dataframes[i][check_feat]
        df_B = dataframes[i+1][check_feat]
        flag = flag and df_A.equals(df_B)
        
    if not flag:
        raise ValueError('Dataframes have different features')
    else:
        return weighted_mean_np