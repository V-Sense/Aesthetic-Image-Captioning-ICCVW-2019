import pickle
import json
import io
import pdb
from tqdm import tqdm
import itertools
import numpy as np

input_json = '/home/ghosalk/Datasets/AVA+PCCD/AVA_TRAINING_FULL_CLEAN.json'
output_json = '/home/ghosalk/Datasets/AVA+PCCD/AVA_TRAINING_FULL_DELIMITED_LDA_LABELLED.json'
lda_path = '/home/ghosalk/Datasets/AVA+PCCD/LDA_AVA_200_50_passes_10000_iter_.p'

ldamodel, dictionary = pickle.load(open(lda_path, 'r'))
db = json.load(io.open(input_json, 'r', encoding = 'utf-8'))
imgs = db['images']

def collect_tokens(comment):
    
    unigrams = comment['unigrams']
    bigrams = comment['bigrams']
    filtered_tokens = [[unigram] for unigram in unigrams] + bigrams
    all_tokens = [' '.join([i[0] for i in n_grams]) for n_grams in filtered_tokens]
    return all_tokens


text_corpus_clean = []
pdb.set_trace()
for count, img in enumerate(tqdm(imgs,  position=0, leave=True, unit=' images')):
    comments = img['sentences']
    all_tokens = map(collect_tokens, comments)
    text_corpus_clean.append(list(itertools.chain.from_iterable(all_tokens)))
    #list_of_clean_tokens += list(itertools.chain.from_iterable(all_tokens))
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_corpus_clean]
#topics_each_image_1 = np.argsort(np.array(ldamodel.get_document_topics(doc_term_matrix, minimum_probability = 0))[:,:,1], axis = 1)[:,-10:]
#lda_labels = np.array(ldamodel.get_document_topics(doc_term_matrix, minimum_probability = 0))[:,:,1]
pdb.set_trace()
for c, (img, tokens) in enumerate(tqdm(zip(imgs, text_corpus_clean),  position=0, leave=True, unit=' images')):
    #img['LDA_LABELS'] = lda_labels[c].tolist()
    #pdb.set_trace()
    img['LDA_LABELS'] = np.array(ldamodel.get_document_topics(dictionary.doc2bow(tokens), minimum_probability = 0))[:,1].tolist()
    
final_db = {}
final_db['images'] = imgs
final_db['dataset'] = "AVA TRAINING AVA+PCCD VALIDATION" 
#pdb.set_trace()
f = io.open(output_json, 'w', encoding = 'utf-8')
f.write(unicode(json.dumps(final_db, ensure_ascii=False)))