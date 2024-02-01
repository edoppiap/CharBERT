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

class IrEvaluator:
    def __init__(self,
                repo_user_prj, #example: 'reingart/pyafipws'
                embedder = 'charbert', #charbert or bert
                root_git_clone = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/git_clones/',
                
                ):


        #Download repo
        self.root_git_clone = Path(root_git_clone)
        self.repo_user_prj = repo_user_prj
        self.prj_name = repo_user_prj.split('/')[-1]
        self.repo_path = root_git_clone / self.prj_name
        
        self.llm = None

        if not os.path.exists(self.repo_path):
            self.repo_path.mkdir()
            print(f'Downloading the repo at {self.repo_path}')
            Repo.clone_from('https://github.com/'+repo_user_prj, to_path = self.repo_path)
        else:
            print(f'Repo already downloaded at {self.repo_path}')

        
        #Create the sentence Transformer
        if embedder == 'bert':
            self.embeddings = HuggingFaceEmbeddings(model_name_or_path = 'bert-base-uncased')
            
        elif embedder == 'charbert':
            charBertTransformer = CharBertTransformer(model_type = 'bert',
                                                    model_name_or_path = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/models/charbert-bert-wiki',
                                                    char_vocab = '/content/drive/MyDrive/PoliTo/NLP_Polito/progetto/codice/CharBERT/data/dict/bert_char_vocab'
                                                        )
            pooling_model = models.Pooling(charBertTransformer.get_word_embedding_dimension()*2)
            sentTrans = SentenceTransformer(modules=[charBertTransformer, pooling_model])
            self.embeddings = CharBertEmbeddings(sentTrans)
        
        
    def create_db(self, chunk_size = 2000, chunk_overlap = 200):
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
        
        self.db = Chroma.from_documents(texts, self.embeddings)
    
    def create_retriever(self, k = 5, search_type = "mmr"):
        #u can create different retrievers without having to recreate the db
        self.retriever = self.db.as_retriever(search_type = search_type, search_kwargs = {"k": k})
    
    def load_dataset(self):
        dataset = load_dataset('Nan-Do/code-search-net-python', split = "train")
        interesting_feat = ['repo', 'path', 'func_name', 'summary']
        dataset = dataset.select_columns(column_names=interesting_feat)

        df = dataset.to_pandas()
        self.df = df[df.repo == self.repo_user_prj]
        print(f'Dataset loaded: {len(self.df)} rows')
        
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
        
    def evaluate_retriever_via_df(self, code_query = False): #TODO vedere se si può passare da qui similarity e k
        k = self.retriever.search_kwargs["k"]
        num_total = len(self.df)
        
        summary = {
            'right_idx': [],
            'wrong_idx': [],
            'top1' : 0,
            'top2' : 0,
            'top3' : 0,
            'top4' : 0,
            'top5' : 0,
        }
        
        for idx, row in self.df.iterrows():
            #print(f'{row.summary = }')
            
            query = row.summary
            
            if code_query:
                query = self.generate_code_query(query)
                
            hits = self.retriever.get_relevant_documents(query)
            preds = [hits[i].metadata['source'].replace(self.repo_path + '/', '') for i in range(k)]
            #print(f'{preds = }')
            #print(f'{row.path=}')
            try:
                match_idx = preds.index(row.path)
                summary['right_idx'].append(idx)
                for j in range(match_idx, k):
                    summary['top'+str(j+1)] += 1/num_total
            except:
                print(f'{preds = }')
                print(f'{row.path=}\n\n')

                summary['wrong_idx'].append(idx)
            
        return summary