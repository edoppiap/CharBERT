from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter

#CharBert
from modeling.modeling_charbert import CharBertTransformer
from sentence_transformers import models, SentenceTransformer
from modeling.charbert_embeddings import CharBertEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

#Chroma
from langchain_community.vectorstores import Chroma

#others
import os
from pprint import pprint
from git import Repo
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import re

from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai

class IrRepoEvaluator:
    def __init__(self,
                repo_user_prj, #example: 'reingart/pyafipws'
                root_git_clone = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/git_clones/',
                root_charbert = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/',
                root_data = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/data/wrap_IR',
                override_df = False
                ):


        #Download repo
        self.root_git_clone = Path(root_git_clone)
        self.repo_user_prj = repo_user_prj
        self.prj_name = repo_user_prj.split('/')[-1]
        self.repo_path = self.root_git_clone / self.prj_name
        self.root_charbert = root_charbert
        self.root_data = root_data
                
        self.llm = None
        
        self.override_df = override_df

        #create the repo folder if not exists
        if not os.path.exists(self.repo_path):
            self.repo_path.mkdir()
            print(f'Downloading the repo at {self.repo_path}')
            Repo.clone_from('https://github.com/'+repo_user_prj, to_path = self.repo_path)
        else:
            print(f'Repo already downloaded at {self.repo_path}')

        
    def get_embeddings(self, embedder):
        #Create the sentence Transformer
        if embedder == 'bert':
            embeddings = HuggingFaceEmbeddings(model_name = 'bert-base-uncased')
            
        elif embedder == 'charbert':
            charBertTransformer = CharBertTransformer(model_type = 'bert',
                                                    model_name_or_path = os.path.join( self.root_charbert ,'models/charbert-bert-wiki'),
                                                    char_vocab =  os.path.join( self.root_charbert , 'data/dict/bert_char_vocab')
                                                    )
            pooling_model = models.Pooling(charBertTransformer.get_word_embedding_dimension()*2)
            sentTrans = SentenceTransformer(modules=[charBertTransformer, pooling_model])
            embeddings = CharBertEmbeddings(sentTrans)

        return embeddings
    
    
    def get_db(self, embeddings, chunk_size = 2000, chunk_overlap = 0):
        #Create the loader
        loader = GenericLoader.from_filesystem(
                path = self.repo_path,
                glob="**/*",
                suffixes=[".py"], #puoi scegliere quali file includere
                #exclude = ["**/non-utf8-encoding.py"], #quali file escludere
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=0),
                #nel parser puoi selezionare il linguaggio e il numero di righe minime che vengono prese per ogni blocco
                )
        
        #load the data
        docs = loader.load()
        print(f'Loader: {len(docs)} chunks')
        
        pattern = r'(?:def|class)\s+(\w+)\s*'

        i_to_del = []
        for i, doc in enumerate(docs):
            matches = re.findall(pattern, doc.page_content)
            doc.metadata['wrap_name'] = matches[0]
            if doc.metadata['content_type'] == 'simplified_code':
                i_to_del.append(i)
                
        for ix in reversed(i_to_del):
            docs.pop(ix)
        
        print(f'Loader after removal: {len(docs)} chunks')

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )
        texts = python_splitter.split_documents(docs)
        print(f'Splitter: {len(texts)} chunks')
        
        db = Chroma.from_documents(texts, embeddings)
        return db
    
    """def create_retriever(self, search_type = "mmr", k = 10):
        #u can create different retrievers without having to recreate the db
        self.retriever = self.db.as_retriever(search_type = search_type, search_kwargs = {"k": k})"""
    
    def load_and_get_dataset_as_df(self):
        dataset = load_dataset('Nan-Do/code-search-net-python', split = "train")
        interesting_feat = ['repo', 'path', 'func_name', 'summary']
        dataset = dataset.select_columns(column_names=interesting_feat)
        
        df = dataset.to_pandas()
        
        df = df[df.repo == self.repo_user_prj]
        df = self.gen_all_code_query(df)
        if self.override_df:
            df.to_csv(f'{self.root_data}/{self.prj_name}.csv', index = False)
            print(f'Dataset saved at: {self.root_data}/{self.prj_name}.csv')
            
        elif not self.override_df and os.path.exists(f'{self.root_data}/{self.prj_name}.csv'):
            raise Exception(f'File {self.root_data}/{self.prj_name}.csv already exists, set override_df = True to overwrite it')
        
        elif not self.override_df and not os.path.exists(f'{self.root_data}/{self.prj_name}.csv'):
            df.to_csv(f'{self.root_data}/{self.prj_name}.csv', index = False)
            print(f'Dataset saved at: {self.root_data}/{self.prj_name}.csv')    
        
        print(f'Dataset loaded: {len(df)} rows.\nCols names: {df.columns}')
        return df
        
    def init_llm(self):
        #Inizializza il llm
        os.environ["GOOGLE_API_KEY"] = 'AIzaSyDVRweStnbEJUjEAV9Mah2ZhEUp2kz0w2M'
        self.llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    def gen_single_code_query(self, query):
        #Questo metodo deve prendere una query in linguaggio naturale
        # e restituire una query in linguaggio di programmazione in formato stringa
        if self.llm is None:
            self.init_llm()
        context = """You are an AI assiantant that converts natural language in python code, return only code in your answer. This is the question: """
        return self.llm.invoke(context+query).replace("```python\n", "").replace("\n```","").replace("```python \n","").replace("```python  \n","")
    
    def gen_all_code_query(self, df):
        tqdm.pandas(desc = "Creating code queries")
        df['code_query'] = df.summary.progress_apply(self.gen_single_code_query)
        return df
        
    def evaluate_db_via_df(self, db, df, search_type, code_query: bool = None):
        #you can pass here search type and k to avoid calling create_retriever to modify the retriever
        # however if must be called before this method the first time
        k = 10
        num_total = len(df)
        
        retriever = db.as_retriever(search_type = search_type, search_kwargs = {"k": k})
        
        summary = {
            'right_idx': [],
            'wrong_idx': [],
            'top_at' : [0]*k,
        }
        
        for idx, row in tqdm(df.iterrows(), total = df.shape[0], desc = 'Evaluating', position = 0, leave = True):
            #print(f'{row.summary = }')
            
            if code_query:
                query = row.code_query
            else:
                query = row.summary
            
            hits = retriever.get_relevant_documents(query)
            preds = [(hits[i].metadata['source'].replace(str(self.repo_path) + '/', ''),
                            hits[i].metadata['wrap_name']) for i in range(k)]
            #print(f'{preds = }')
            #print(f'{row.path=}')
            label = (row.path, row.func_name)
            try:
                match_idx = preds.index(label)
                summary['right_idx'].append(idx)
                for j in range(match_idx, k):
                    summary['top_at'][j] += 1/num_total
            except:
                #print(f'{preds = }')
                #print(f'{row.path=}\n\n')
                summary['wrong_idx'].append(idx)
                
        return summary
    
    def main(self):
        result_df = pd.DataFrame()
        df = self.load_and_get_dataset_as_df()
        
        for embedder in tqdm(['charbert', 'bert'], desc='Embedder'): #for each embedder
            print(f'\nEmbedder: {embedder}')
            embeddings = self.get_embeddings(embedder)
            
            for chunk_size in [1000, 2000, 3000]: #for each chunk size
                print(f'\nChunk size: {chunk_size}')
                #Quando crei un nuovo db devi cancellare quello vecchio, altrimenti si aggiungono
                if 'db' in locals():
                    print('Deleting old db')
                    db.delete_collection()
                    
                db = self.get_db(embeddings = embeddings, chunk_size = chunk_size)
                print(f"Num_chunks in db {len(db.get()['ids'])}")
                
                for code_query in [True, False]: #for code query or not
                    print(f'\nCode query: {code_query}')
                    
                    for search_type in ['mmr', 'similarity']: #for each search type
                        summary = self.evaluate_db_via_df(db, df, search_type = search_type, code_query = code_query)
                        #change format
                        summary['right_idx'] = [summary['right_idx']]
                        summary['wrong_idx'] = [summary['wrong_idx']]
                        #Salva ogni top in una colonna diversa
                        summary['top_at'] = [np.array(summary['top_at'])]
                        
                        #attach run info to summary
                        summary['embedder'] = embedder
                        summary['chunk_size'] = chunk_size
                        summary['code_query'] = code_query
                        summary['search_type'] = search_type
                        summary['prj_name'] = self.prj_name
                        summary['num_queries'] = len(df)
                        
                        print(f"{summary['embedder']=}")
                        print(f"{summary['chunk_size']=}")
                        print(f"{summary['code_query']=}")
                        print(f"{summary['search_type']=}")
                        print(f"{summary['top_at'] = }\n")
                        
                        tmp_df = pd.DataFrame(summary)
                        result_df = pd.concat([result_df, tmp_df])
                        
                #result_df.to_csv(f'Ir_wrap_results/{self.prj_name}.csv', index = False)
                result_df.to_json(f'outputs/Ir_wrap_results/{self.prj_name}.json', orient='records')
        
        print(f'outputs/Ir_wrap_results/{self.prj_name}.json')
        print('Done')