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

class IrEvaluator:
    def __init__(self,
                repo_user_prj, #example: 'reingart/pyafipws'
                root_git_clone = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/git_clones/',
                ):


        #Download repo
        self.root_git_clone = Path(root_git_clone)
        self.repo_user_prj = repo_user_prj
        self.prj_name = repo_user_prj.split('/')[-1]
        self.repo_path = self.root_git_clone / self.prj_name
                
        self.llm = None

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
            embeddings = HuggingFaceEmbeddings(model_name_or_path = 'bert-base-uncased')
            
        elif embedder == 'charbert':
            charBertTransformer = CharBertTransformer(model_type = 'bert',
                                                    model_name_or_path = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/models/charbert-bert-wiki',
                                                    char_vocab = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/data/dict/bert_char_vocab'
                                                        )
            pooling_model = models.Pooling(charBertTransformer.get_word_embedding_dimension()*2)
            sentTrans = SentenceTransformer(modules=[charBertTransformer, pooling_model])
            embeddings = CharBertEmbeddings(sentTrans)

        return embeddings
    
    
    def get_db(self, chunk_size = 2000, chunk_overlap = 200):
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

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )
        texts = python_splitter.split_documents(docs)
        print(f'Splitter: {len(texts)} chunks')
        db = Chroma.from_documents(texts, self.embeddings)
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
        print(f'Dataset loaded: {len(self.df)} rows')
        return df
        
    def init_llm(self):
        #Inizializza il llm
        self.llm = ...
    
    def generate_code_query(self, query):
        #Questo metodo deve prendere una query in linguaggio naturale
        # e restituire una query in linguaggio di programmazione in formato stringa
        if self.llm is None:
            self.init_llm()
        
        pass
        return None
        
    def evaluate_db_via_df(self, db, df, search_type, code_query: bool = None):
        #you can pass here search type and k to avoid calling create_retriever to modify the retriever
        # however if must be called before this method the first time
        k = 10
        num_total = len(df)
        
        retriever = db.as_retriever(search_type = search_type, search_kwargs = {"k": k})
        
        summary = {
            'right_idx': [],
            'wrong_idx': [],
            'top@' : [0]*k,
        }
        
        for idx, row in df.iterrows():
            #print(f'{row.summary = }')
            
            query = row.summary
            
            if code_query:
                query = self.generate_code_query(query)
            
            hits = self.retriever.get_relevant_documents(query)
            preds = [hits[i].metadata['source'].replace(str(self.repo_path) + '/', '') for i in range(k)]
            #print(f'{preds = }')
            #print(f'{row.path=}')
            try:
                match_idx = preds.index(row.path)
                summary['right_idx'].append(idx)
                for j in range(match_idx, k):
                    summary['top@'][j] += 1/num_total
            except:
                #print(f'{preds = }')
                #print(f'{row.path=}\n\n')
                summary['wrong_idx'].append(idx)
                
        return summary
    
    def main(self):
        result_df = pd.DataFrame()
        df = self.load_and_get_dataset_as_df()
        
        for embedder in tqdm(['charbert', 'bert'], desc='Embedder'): #for each embedder
            print(f'Embedder: {embedder}')
            embeddings = self.get_embeddings(embedder)
            
            for chunk_size in [1000, 2000, 3000]: #for each chunk size
                print(f'\tChunk size: {chunk_size}')
                db = self.get_db(chunk_size = chunk_size)
                
                for code_query in [False, True]: #for code query or not
                    print(f'\t\tCode query: {code_query}')
                    
                    for search_type in ['mmr', 'similarity']: #for each search type
                        summary = self.evaluate_db_via_df(db, df, search_type = search_type, code_query = code_query)
                        #attach run info to summary
                        summary['embedder'] = embedder
                        summary['chunk_size'] = chunk_size
                        summary['code_query'] = code_query
                        summary['search_type'] = search_type
                        pprint(summary)
                        tmp_df = pd.DataFrame(summary)
                        result_df = pd.concat([result_df, tmp_df])
                        
                result_df.to_csv('result_df.csv', index = False)
                break
                        