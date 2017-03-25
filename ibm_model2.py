from collections import Counter
import codecs
import itertools
from collections import defaultdict
import math
import pickle

### FILE PATHS
train_en_path = '/home/ubuntu/11-731/data/en-de/train.en-de.low.filt.en'
train_de_path = '/home/ubuntu/11-731/data/en-de/train.en-de.low.filt.de'

valid_en_path = '/home/ubuntu/11-731/data/en-de/valid.en-de.tok.en'
valid_de_path = '/home/ubuntu/11-731/data/en-de/valid.en-de.tok.de'

class Corpus(object):
    def __init__(self, filename, target=False):
        with codecs.open(filename, 'r') as inp:
            data = inp.readlines()
            
        data = [x.strip().lower() for x in data]
        all_wrds = []
        self.sentences = []
        for sent in data:
            wrds = sent.split(' ')
            wrds = filter(None, wrds)
            if target:
                wrds.append('NULL')
            all_wrds.extend(wrds)
            self.sentences.append(wrds)
        self.vocab = Counter(all_wrds)
    

def ibm_model2(all_tuples, source_wrds, target_wrds, no_epochs):
    
    model2_prob = defaultdict(float)

    for no in range(len(source_wrds.sentences)):
        source_len = len(source_wrds.sentences[no])
        target_len = len(target_wrds.sentences[no])

        for j in range(source_len):
            for i in range(target_len):
                model2_prob[(i, j, source_len, target_len)] = 1/float(target_len)

    for epoch in range(no_epochs):
        
        count_tgs = defaultdict(float)
        count_s = defaultdict(float)
        count_si_ti = defaultdict(float)
        count_ti = defaultdict(float)
        
        for no in range(len(source_wrds.sentences)):
            
            source_len = len(source_wrds.sentences[no])
            target_len = len(target_wrds.sentences[no])
            
            for j, e_wrd in enumerate(source_wrds.sentences[no]):
                curr_total = sum(all_tuples[(e_wrd, f_wrd)] * model2_prob[(i, j, source_len, target_len)] for i, f_wrd in enumerate(target_wrds.sentences[no]))
                
                for i, f_wrd in enumerate(target_wrds.sentences[no]):
                    tmp = all_tuples[(e_wrd, f_wrd)] * (model2_prob[(i, j, source_len, target_len)]/float(curr_total))
                    #except:
                    #print all_tuples[(e_wrd, f_wrd)], model2_prob[(i, j, source_len, target_len)], float(curr_total), target_len
                    count_tgs[(e_wrd, f_wrd)] += tmp
                    count_s[e_wrd] += tmp
                    count_si_ti[(i, j, source_len, target_len)] += tmp
                    count_ti[(j, source_len, target_len)] += tmp
                    
        for e_wrd, f_wrd in count_tgs:
            all_tuples[(e_wrd, f_wrd)] = count_tgs[(e_wrd, f_wrd)]/float(count_s[e_wrd])

        for i, j, source_len, target_len in count_si_ti:
            model2_prob[(i, j, source_len, target_len)] = count_si_ti[(i, j, source_len, target_len)]/float(count_ti[(j, source_len, target_len)])
            
    return all_tuples

def get_alignments(source_wrds, target_wrds, all_tuples):
    ### Get alignments
    alignments = []
    for i in range(len(source_wrds.sentences)):
        curr_align = []
        for f, f_wrd in enumerate(target_wrds.sentences[i]):
            tmp = [(x, f_wrd) for x in source_wrds.sentences[i]]
            prob = [all_tuples[x] for x in tmp]
            index = prob.index(max(prob))
            align = str(index) + '-' + str(f)
            curr_align.append(align)
        alignments.append(' '.join(curr_align))
        
    with open("alignments_ibm2_sam_5.txt", "w") as out:
        for i in range(len(alignments)):
            out.write(alignments[i] + "\n")
        
    return alignments

def test():
    source_wrds = Corpus(train_en_path)
    target_wrds = Corpus(train_de_path)

    with open("tuples_sam.pkl", "r") as inp:
        all_tuples = pickle.load(inp)

    no_epochs = 5
    mod_tuples = ibm_model2(all_tuples, source_wrds, target_wrds, no_epochs)

    with open("tuples_ibm2_sam_5.pkl", "w") as out:
        pickle.dump(mod_tuples, out)
    alignments = get_alignments(source_wrds, target_wrds, mod_tuples)
    
def run_ibm2(source_wrds, target_wrds, all_tuples, no_epochs):
    mod_tuples = ibm_model2(all_tuples, source_wrds, target_wrds, no_epochs)
    return mod_tuples
        