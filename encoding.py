import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from queue import PriorityQueue
import random
from gensim.utils import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cosine

def semantic_classes(lyrics, class_list, num_classes=12, device='cpu', strategy='random'):
    transform = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    with open(lyrics) as lyrics_file:
        lines = lyrics_file.readlines()
        lines = [line.rstrip() for line in lines if not re.match('\[.*\]$', line.rstrip())]
    if strategy == 'best':
        top_keys = PriorityQueue()
    elif strategy == 'random':
        top_keys = set()
    for l in tqdm(lines):
        best_key, best_sim = 0, -1
        for key in range(len(class_list)):
            class_emb = transform.encode(class_list[key], convert_to_tensor=True)
            l_emb = transform.encode(l, convert_to_tensor=True)
            cos_sim = util.pytorch_cos_sim(class_emb, l_emb)
            if cos_sim.item() > best_sim:
                best_key, best_sim = key, cos_sim.item()
        if strategy == 'best':
            top_keys.put((-best_sim, best_key))
        elif strategy == 'random':
            top_keys.add(best_key)
    if strategy == 'best':
        # get num_classes best keys in PriorityQueue
        keys = []
        while len(keys) < num_classes:
            pq_key = top_keys.get()[1]
            if pq_key not in keys: keys.append(pq_key)
    elif strategy == 'random':
        keys = random.sample(top_keys, num_classes)
    return keys

def doc2vec_classes(lyrics, class_list, num_classes=12, strategy='ransac'):
    tagged_l = [TaggedDocument(d, [i]) for i, d in enumerate(doc_tokenize(lyrics))]
    model = Doc2Vec(tagged_l, vector_size=20, window=2, min_count=1, epochs=100)
    lvec = model.infer_vector(' '.join(lyrics).split())
    if strategy == 'ransac':
        best_classes = ransac(class_list, model, num_classes, iter=10000)
    return best_classes

def ransac(clist, lvec, model, num_classes, iter=100):
    best_idx, best_dist = None, float('inf')
    for _ in range(iter):
        csample_idx = random.sample(list(range(len(clist))), num_classes)
        csample = [clist[i] for i in csample_idx]
        cvec = model.infer_vector(csample)
        cdist = cosine(cvec, lvec)
        if cdist < best_dist:
            best_idx, best_dist = csample_idx, cdist
    return best_idx

        

def doc_tokenize(doc):
    return [tokenize(s.lower()) for s in doc]