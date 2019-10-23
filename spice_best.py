from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
from collections import Counter
import pdb

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice_Best:
    """
    Main Class to compute the SPICE metric 
    """

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan
          
    def compute_spice_for_1_ref_caption(self, new_input_data):
            cwd = os.path.dirname(os.path.abspath(__file__))
            temp_dir=os.path.join(cwd, TEMP_DIR)
            if not os.path.exists(temp_dir):
              os.makedirs(temp_dir)
            in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
            json.dump(new_input_data, in_file, indent=2)
            in_file.close()
    
            # Start job
            out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
            out_file.close()
            cache_dir=os.path.join(cwd, CACHE_DIR)
            if not os.path.exists(cache_dir):
              os.makedirs(cache_dir)
            spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
              '-cache', cache_dir,
              '-out', out_file.name,
              #'-threads','2',
              '-subset',
              '-silent'
            ]
            subprocess.check_call(spice_cmd, 
                cwd=os.path.dirname(os.path.abspath(__file__)))
    
            # Read and process results
            with open(out_file.name) as data_file:
              results = json.load(data_file)
            os.remove(in_file.name)
            os.remove(out_file.name)
    
            #imgId_to_scores = {}
            #spice_scores = []
            spice_scores_dict = []
            for item in results:
              #imgId_to_scores[item['image_id']] = item['scores']
              #spice_scores.append(self.float_convert(item['scores']['All']['f']))
              #spice_scores_dict[item['image_id']] = self.float_convert(item['scores']['All']['f'])
               f_Score = self.float_convert(item['scores']['All']['f'])
               precision = self.float_convert(item['scores']['All']['pr'])
               recall = self.float_convert(item['scores']['All']['re'])
               spice_scores_dict.append((item['image_id'], [f_Score, precision, recall]))
               #spice_precision_dict.append((item['image_id'], ))
               #spice_recall_dict.append((item['image_id'], ))
            '''average_score = np.mean(np.array(spice_scores))
            scores = []
            for image_id in imgIds:
              # Convert none to NaN before saving scores over subcategories
              score_set = {}
              for category,score_tuple in imgId_to_scores[image_id].iteritems():
                score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
              scores.append(score_set)'''
            return spice_scores_dict
            
    def duplicate_gt(self, id_gt):
        imid = id_gt[0]
        caps = id_gt[1]
        #pdb.set_trace()
        n_refs = len(caps)
        to_repeat = self.n_max_ref - n_refs
        caps = caps + [caps[-1]] * to_repeat
        return (imid,caps)
        
    
    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        #pdb.set_trace()
        score_dict = dict((key,[]) for key in imgIds)
        best_score_dict = dict((key,[]) for key in imgIds)
        self.n_max_ref = np.max([len(gts[k]) for k in imgIds]) 
        gts = dict(map(self.duplicate_gt, gts.items()))
        #pdb.set_trace()
        # Prepare temp input file for the SPICE scorer
        for i in range(self.n_max_ref):
            
            input_data = []
            for id in imgIds:
                hypo = res[id]
                ref = [gts[id][i]]
    
                # Sanity check.
                assert(type(hypo) is list)
                assert(len(hypo) == 1)
                assert(type(ref) is list)
                assert(len(ref) >= 1)
    
                input_data.append({
                  "image_id" : id,
                  "test" : hypo[0],
                  "refs" : ref
                })
            #pdb.set_trace()
            spice_scores_list = self.compute_spice_for_1_ref_caption(input_data)
            for key,value in spice_scores_list:
                score_dict[key].append(value)
            
        #pdb.set_trace()
        for key, values in score_dict.items():
                arg_max_p = np.argmax([val[0] for val in values])
                best_score_dict[key] = values[arg_max_p]
        #pdb.set_trace()
        average_score = tuple(np.mean(np.array(best_score_dict.values()), axis = 0))
        return average_score, spice_scores_list

    def method(self):
        return "SPICE_BEST"


