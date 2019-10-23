from __future__ import print_function
import json
import io
import pdb
from gensim import corpora
import itertools
from collections import Counter
from nltk.corpus import stopwords
import gensim
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import visdom
import os
from random import shuffle
from torchvision import transforms
from PIL import Image
import torch
all_tokens = []

comment_file = 'CLEAN_AVA_FULL_AFTER_SUBJECTIVE_CLEANING.json'
img_src = '/home/koustav/Datasets/AVADataSet/'
db = json.load(io.open(comment_file, 'r', encoding = 'utf-8'))
imgs = db['images']
lda_vocab = io.open('LDA_VOCAB.txt','w', encoding = 'utf-8')
discarded_uni = io.open('Discarded_Unigrams.txt','w', encoding = 'utf-8')
discarded_bi = io.open('Discarded_Bigrams.txt','w', encoding = 'utf-8')
stop = (set(stopwords.words('english')) | set(['.','?', '!', ','])) - set(['above', 'below', 'before', 'after', 'too', 'very'])
discarded_uni_list = []
discarded_bi_list = []
lemmatizer = WordNetLemmatizer()

def prepare_visuals(image_list):
    topic, image_list = image_list

    def splitter(n, s):
        pieces = s.split(' ')            
        return(" ".join(pieces[i:i+n]) for i in xrange(0, len(pieces), n))

    global vis_1
    transform_to_tensor = transforms.ToTensor()
    T_tensor = []
    caption = 'Topic Number : ' + unicode(topic) + ' '
    for c, img in enumerate(image_list):
        I = Image.open(img[0]).convert('RGB')
        caption += '( ' + unicode(c) + ' ) ' + img[1] + ' '
        T_tensor.append(transform_to_tensor(I.resize([128,128])))            
    vis_1.images(torch.stack(T_tensor, 0), opts = {'nrow' : len(image_list) + 2, 'caption' : caption})

def filter_tokens(poss):
    global discarded_uni_list, discarded_bi_list
    n_flag = False
    if len(poss) == 1:
        if poss[0][1] in ['NOUN']:
            n_flag = True
        if not n_flag:
            discarded_uni_list+= [poss[0][0]]
        return n_flag
    else:
        if poss[0][1] in ['NOUN', 'ADJ', 'ADV'] and poss[1][1] in ['NOUN', 'ADJ']:
            n_flag = True
        if not n_flag:
            discarded_bi_list += [poss[0][0] + ' ' + poss[1][0]]
        return n_flag
        
def lemmatize(pos):
    global lemmatizer
    if pos[1] == 'NOUN':
        return (lemmatizer.lemmatize(pos[0], wordnet.NOUN), pos[1])
    elif pos[1] == 'VERB':
        return (lemmatizer.lemmatize(pos[0], wordnet.VERB), pos[1])
    elif pos[1] == 'ADJ':
        return (lemmatizer.lemmatize(pos[0], wordnet.ADJ), pos[1])        
    elif pos[1] == 'ADV':
        return (lemmatizer.lemmatize(pos[0], wordnet.ADV), pos[1])
    else:
        return pos

def collect_tokens(comment):
    unigrams = comment['unigrams']
    bigrams = comment['bigrams']
    filtered_tokens = [[unigram] for unigram in unigrams] + bigrams
    all_tokens = [' '.join([i[0] for i in n_grams]) for n_grams in filtered_tokens]
    return all_tokens
    
def filter_comments_on_count(comment):
    global cw
    new_comment = []
    for c in comment:
        if (len(c.split()) == 1 and cw[c] > 5) or (len(c.split()) == 2 and cw[c] > 5): # only if unigram count is more than 5 and bigram count is more than 3
            new_comment.append(c)    
    return new_comment


# Use/Modify this function depending on the strategy of how the CNN is trained. Move this to a different script if more convenient
def find_topic(text_corpus_clean_new, ldamodel, num_topics, topics_each_image_1):
    global topics_word_list, topic_probs, topic_words
    topic_labels = []
    topics_word_list = ldamodel.print_topics(num_topics = num_topics, num_words=25)
    topic_words = [[i.split('*')[1].strip().strip('\"') for i in temp_topic_words[1].split('+')] for temp_topic_words in topics_word_list]
    topic_probs = [[float(i.split('*')[0].strip()) for i in temp_topic_words[1].split('+')] for temp_topic_words in topics_word_list]
    for tokens, topics in zip(text_corpus_clean_new, topics_each_image_1):
        this_topic_words = [topic_words[i] for i in topics]
        n_matches = [len(set(tokens) & set(words)) for words in this_topic_words]
        labels = [topics[i] for i in  np.where(np.array(n_matches) == np.max(n_matches))[0]]
        if len(labels) > 1:
            first_w_probs = []
            for lab in labels:
                first_w_probs.append(topic_probs[lab][0])
            topic_labels.append(labels[np.argmax(first_w_probs)])
        else:
            topic_labels.append(labels[0])                
    return topic_labels
    
def filter_topics (img_topic):
    global ldamodel, topics_word_list, topic_probs
    img, topic = img_topic
    if topic_probs[topic][0] > 0.05:
        img['Label_LDA'] = topic
        return True
    else:
        return False

text_corpus_clean = []
list_of_clean_tokens = []
for count, img in enumerate(imgs):
    if count % 1000 == 0:
        print ('%d / %d images processed'%(count, len(imgs)))        
    comments = img['sentences']
    all_tokens = map(collect_tokens, comments)
    text_corpus_clean.append(all_tokens)
    list_of_clean_tokens += list(itertools.chain.from_iterable(all_tokens))

lda_vocab.write("\n".join([unicode(i) + ' '+ unicode(j) for i,j in Counter(list_of_clean_tokens).most_common()]))
cw = Counter(list_of_clean_tokens)
text_corpus_clean_new = []

for count_1, comments in enumerate(text_corpus_clean):
    #text_corpus_clean_new.append(list(itertools.chain.from_iterable(map(filter_comments_on_count, comments))))
    text_corpus_clean_new.append(list(itertools.chain.from_iterable(comments)))

dictionary = corpora.Dictionary(text_corpus_clean_new)
dictionary.filter_extremes(no_below=30, no_above=0.10)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_corpus_clean_new]
print ('dictionary shape : %d \ndoc_term_matrix shape : %d'%(len(dictionary), len(doc_term_matrix)))
print('Starting LDA')
Lda = gensim.models.ldamulticore.LdaMulticore
for num_topics in range(200,201)[::20]:
    vis_1 = visdom.Visdom( env = 'Topic-Visualization_After_New_Subjectivity_'+ str(num_topics))
    vis_1.close()
    ldamodel = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary, passes = 200, workers  = 15, iterations = 5000, chunksize = 20000 )
    cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass')
    cm_score = cm.get_coherence()
    #pdb.set_trace()    
    topics_each_image_1 = np.argsort(np.array(ldamodel.get_document_topics(doc_term_matrix, minimum_probability = 0))[:,:,1], axis = 1)[:,-10:]
    topics_each_image_2 = find_topic(text_corpus_clean_new, ldamodel, num_topics, topics_each_image_1)
    #pdb.set_trace()
    im_info = [(os.path.join(img_src, img['filename']),' || '.join([sent['clean'] for sent in img['sentences']])) for img in imgs]
    topic_counter = Counter(topics_each_image_2).most_common()
    image_list = []
    for topic, count in topic_counter:
        #pdb.set_trace()
        indices = np.where(topics_each_image_2 == topic)[0]
        shuffle(indices)
        indices = indices[0:16]
        image_list.append((topic, [im_info[i] for i in indices]))
    #pdb.set_trace()
    #pdb.set_trace()
    topics = ldamodel.print_topics(num_topics = num_topics, num_words=25)
    topic_summaries = [unicode(t) + ' ' + unicode(c) + ' '+ unicode(topics[t][1]) for t,c in topic_counter]
    print ('%d : %f'%(num_topics, cm_score))
    with io.open('Iterative_LDA/_temp_topics_iteration_'+str(num_topics)+'.txt','w', encoding = 'utf-8') as f1:
        print (unicode(cm_score), file = f1)
        print('\n'.join(topic_summaries), file= f1)
    map(prepare_visuals, image_list)
    new_imgs = [img[0] for img in filter(filter_topics, zip(imgs, topics_each_image_2))]
    labels = [img['Label_LDA'] for img in new_imgs]
    label_hash =  dict(zip(Counter(labels).keys(),range(len(Counter(labels).keys()))))
    for img in new_imgs: img['Label_LDA'] = label_hash[img['Label_LDA']]
    pdb.set_trace()
#with open('Iterative_LDA/_temp_topics_iteration_'+str(iteration)+'.txt','w') as f1:
#    print('\n'.join(map(str,ldamodel.print_topics(num_topics = 50, num_words=20))), file= f1)
    #print(Counter(np.argmax(prob_dist,axis = 1).tolist()), file= f1)
#pickle.dump([ldamodel,dictionary], open('Iterative_LDA/Models/LDA_AVA' + '50' + '_' + str(iteration) + '.p','w'))
#pdb.set_trace()