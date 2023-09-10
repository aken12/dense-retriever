import torch.nn.functional as F

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import numpy as np
import faiss
import csv
import argparse
from tqdm import tqdm

csv.field_size_limit(500 * 1024 * 1024) 

class E5ReteriveModel():
    def __init__(self,args):
        self.model_name = args.model_path if args.model_name is None else args.model_name
        self.model,self.tokenizer = self.load_e5_model(self.model_name)
        self.d = args.d
        self.res = faiss.StandardGpuResources()
        self.model = self.model.cuda()
        self.result = {}
        self.doc_id_path = args.doc_id_path
        self.doc_id_dict = {}
        self.num_lines = sum(1 for line in open(args.docs_path))
        with open(self.doc_id_path) as fr:
            reader = csv.reader(fr, delimiter='\t')
            for line in reader:
                self.doc_id_dict[int(line[0])] = line[1]
        self.count = 0

    def average_pool(self,
                    last_hidden_states,attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def generate_e5_embs(self,
                        input_texts):
        with torch.no_grad():
            batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def caluclate_e5_score(self,
                           queries,passages):
        q_embs = self.generate_e5_embs(queries)
        p_embs = self.generate_e5_embs(passages)
        scores = (q_embs @ p_embs[2:].T) * 100
        return scores.tolist()

    def caluclate_e5_score_np(self,
                              queries,
                              passages):
        passages = np.array(passages)
        scores = np.dot(queries,passages.T)
        return scores

    def load_e5_model(self,
                      model_name): #'intfloat/multilingual-e5-large'
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model,tokenizer

    def docs_batch_loader(self,docs_path,block_size):
        per_block_passages = self.num_lines // block_size
        ids_map = []
        passages = []
        with open(docs_path)as f:
            reader = csv.reader(f,delimiter='\t')
            for line in reader:
                ids_map.append(line[0])
                passages.append(f'passage: + {line[1]}')
                if len(ids_map) == per_block_passages:
                    yield ids_map,passages
                    ids_map = []
                    passages = []
            if ids_map:
                yield ids_map,passages

    def load_queries(self,query_path):
        ids = []
        input_texts = []
        with open(query_path)as fr:
            reader = csv.reader(fr,delimiter='\t')
            for line in reader:
                ids.append(line[0])
                input_texts.append(f'query: + {line[1]}')
            q_embs = self.generate_e5_embs(input_texts)
            q_embs = q_embs.cpu().numpy()
        return ids,q_embs

    def e5batchindex(self,input_texts,queries,ids=None, batch_size=256):
        self.count += 1
        print(f"{self.count} indexing")
        index = faiss.IndexFlatL2(self.d)
        n = len(input_texts)
        self.embeddings = np.zeros((n, self.d))
        for i in tqdm(range(0, n, batch_size),total=int(n//batch_size)):
            end = min(i + batch_size, n)
            batch_texts = input_texts[i:end]
            batch_embs = self.generate_e5_embs(batch_texts)
            self.embeddings[i:end] = batch_embs.cpu().numpy()
        index.add(self.embeddings)
        distances, indices = index.search(queries, 1000)
        index.reset()
        return distances,indices

    def score_ranking(self,distances,indices,ids):
        for i in range(indices.shape[0]):
            qid = ids[i]
            for j in range(indices.shape[1]):
                doc_id = indices[i, j]
                doc_id_str = self.doc_id_dict[doc_id]
                score = -distances[i, j]  # Use negative distances as scores
                rank = j + 1
                result_line = [qid, "Q0", doc_id_str, rank, score, "E5_retriever"]
                if not qid in self.result.keys():
                    self.result[qid] = []
                else:
                    self.result[qid].append(result_line)
        return
    
def main():
    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--d', dest='d',default=None,type=int)
    parser.add_argument('--docs_path', dest='docs_path',default=None,type=str)
    parser.add_argument('--model_path', dest='model_path',default=None,type=str)
    parser.add_argument('--model_name', dest='model_name',default=None,type=str)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int)
    parser.add_argument('--block_size', dest='block_size',default=None,type=int)
    parser.add_argument('--query_path', dest='query_path',default=None,type=str)
    parser.add_argument('--doc_id_path', dest='doc_id_path',default=None,type=str)
    parser.add_argument('--result_path', dest='result_path',default=None,type=str)

    args = parser.parse_args()
    e5model = E5ReteriveModel(args)
    ids,queries = e5model.load_queries(query_path=args.query_path)
    id_tuning = 0

    for batch_ids,batch_texts in tqdm(e5model.docs_batch_loader(block_size=args.block_size,docs_path=args.docs_path),total=args.block_size):
        distances,indices = e5model.e5batchindex(input_texts=batch_texts,queries=queries,ids=batch_ids,batch_size=args.batch_size)
        indices += id_tuning
        id_tuning += len(batch_texts)
        e5model.score_ranking(indices=indices,distances=distances,ids=ids)

    with open(args.result_path,"w")as fw:
        writer = csv.writer(fw,delimiter=' ')
        for qid, results in e5model.result.items():
            sorted_results = sorted(results, key=lambda x: x[4], reverse=True)
            e5model.result[qid] = sorted_results[:1000]
            for line in e5model.result[qid]:
                writer.writerow(line)

if __name__=='__main__':
    main()