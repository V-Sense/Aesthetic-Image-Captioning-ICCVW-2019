from __future__ import print_function
import json
import pdb
import re
#from scipy.misc import imread, imresize

import numpy as np
from nltk.tokenize import RegexpTokenizer
import io
#from langdetect import detect
#from langdetect import lang_detect_exception
#import string
#from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
#from pattern.en import suggest
from string import digits
from PIL import Image
import os
from langdetect import DetectorFactory
DetectorFactory.seed = 0

tokenizer = RegexpTokenizer(r'\w+\S*\w*')
stop = set(stopwords.words('english'))
#exclude = set(string.punctuation)
#replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

chars = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : '',          # modifier - under line
    '\xc2\xb4' : '\''
    
}

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def replace_chars(match):
    char = match.group(0)
    return chars[char]
    
def clean_string(text):
    low_text = text.lower()
    #punc_free = ' '.join(ch for ch in low_text if ch not in exclude)
    digit_free = low_text.translate(None, digits)
    punc_free = re.sub(ur"[^\w\d'\s]+",'',digit_free)    
    #punc_free = digit_free.translate(replace_punctuation)
    #tokens = [suggest(reduce_lengthening(word))[0][0] if suggest(reduce_lengthening(word))[0][1] >= 0.95 else reduce_lengthening(word) for word in tokenizer.tokenize(punc_free)]
    tokens =   tokenizer.tokenize(punc_free)  
    #tokens = [token if len(token) < 12 else reduce_lengthening(token) for token in tokens]
    #tokens = [token if suggest(token)[0][1] <= 0.95 else suggest(token)[0][0] for token in tokens]
    return ' '.join(w for w in tokens), tokens
    #return re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, punc_free)
    
comment_file = "AVA_Comments_Full.txt"
ava_path = "/home/koustav/Desktop/AVADataSet/"
parent_list = []
num_lines = sum(1 for line in open(comment_file))
f1 = io.open('Non_English.captions','w', encoding = 'utf-8')
f2 = io.open('No_Captions.captions','w',encoding = 'utf-8')
#test_indices = np.random.randint(0, num_lines, 5000)
allot_indices = np.arange(num_lines)
np.random.shuffle(allot_indices)
test_indices = allot_indices[0:5000]
val_indices = allot_indices[5000:10000]
train_indices = allot_indices[10000:]

'''small_indices = test_indices = np.arange(num_lines)
np.random.shuffle(small_indices)
small_indices = small_indices[0:20000]'''


with io.open(comment_file, 'r', encoding = 'utf-8') as f:
    for count, line in enumerate(f) :
        if count % 1000 == 0:
            print("%d / %d files processed"%(count, num_lines))            
        #if (not count in train_indices) and (not count in test_indices) and (not count in val_indices):
        #    continue
        elements = line.strip('\n').split('#')
        img={}
        image_path = elements[1]
        captions = elements[2:]
        
        img[u'filename'] = image_path.strip() + '.jpg'
        try:
            img[u"imgid"] = int(image_path.strip())
        except ValueError:
            continue
        if count in train_indices:
            img[u"split"] = u"train"
        elif count in val_indices:
            img[u"split"] = u"val"
        else:
            img[u"split"] = u"test"
        img[u"cocoid"] = img[u"imgid"]
        img[u"sentids"] = ''
        img[u"filepath"] = ""
            
        sentences = []
        
        for cap in captions:
            #try :
            #    if not detect(mul_cap) == u'en':
            #        print ('%s language : %s'%(detect(mul_cap), mul_cap), file = f1)
            #        #pdb.set_trace()
            #        continue
            #except lang_detect_exception.LangDetectException:
            #    continue                 
            #split_mul_cap = re.split(';|\?|\!|\.', mul_cap)
            #pdb.set_trace()
            
            sentence = {}
            try:
                clean_cap, tokens = clean_string(cap.encode('utf-8'))
                sentence[u'raw'] = cap
                sentence[u'clean'] = clean_cap
            except UnicodeDecodeError:
                print("Bad caption : %s"%(cap.encode('utf-8')))
                continue
                
                
            sentence[u'imgid'] = img["imgid"]
            sentence[u'sentid'] = ''
            sentence[u'tokens'] = tokens
            #if len(sentence['tokens']) heeelosdsad< 1 :
            #    continue
            sentences.append(sentence)
        if len(sentences) == 0:
            #pdb.set_trace()
            print ("%s"%(line), file =f2)
            continue
        img[u"sentences"] = sentences
        
        #pdb.set_trace()
        try:
            image = Image.open(ava_path + img['filename']) # open the image file
            image = image.resize((256,256)) # verify that it is, in fact an image
        except (IOError, SyntaxError, ValueError) as e:
            print('Bad file:', img['filename']) # print out the names of corrupt files
            try:
                os.remove(ava_path + img['filename'])
            except OSError:
                continue
        parent_list.append(img)
        
#pdb.set_trace()
final_db = {}
final_db[u'images'] = parent_list
final_db[u'dataset'] = u'AVA'
pdb.set_trace()
with io.open('CLEAN_AVA_FULL_COMMENTS.json','w', encoding = 'utf-8') as f:
    f.write(unicode(json.dumps(final_db, ensure_ascii=False)))