from collections import Counter
import codecs
import itertools
from collections import defaultdict
import pickle
import sys
from ibm_model2 import *

class Corpus(object):
    def __init__(self, filename):
        with codecs.open(filename, 'r') as inp:
            data = inp.readlines()
            
        data = [x.strip().lower() for x in data]
        all_wrds = []
        self.sentences = []
        for sent in data:
            wrds = sent.split(' ')
            wrds = filter(None, wrds)
            all_wrds.extend(wrds)
            self.sentences.append(wrds)
        self.vocab = Counter(all_wrds)
    
#source_wrds = Corpus(valid_en_path)
#target_wrds = Corpus(valid_de_path)

def create_tuples(source_wrds, target_wrds):
    all_tuples = {}
    foreign = {}
    for i in range(len(source_wrds.sentences)):
        for e_wrd in source_wrds.sentences[i]:
            for f_wrd in target_wrds.sentences[i]:
                all_tuples.setdefault((e_wrd, f_wrd), 0)
                foreign.setdefault(f_wrd, 0)
                
    return all_tuples, foreign

def create_align(no_epochs, all_tuples, foreign, source_wrds, target_wrds):

    for epoch in range(no_epochs):
        print "Started Epoch: " + str(epoch)
        ### Initialization
        count = all_tuples.fromkeys(all_tuples.keys(), 0)
        total = foreign.fromkeys(foreign.keys(), 0)

        print "Starting Normalization and counts"
        for i in range(len(source_wrds.sentences)):
            ### Compute Normalization
            s_total = {}
            for e_wrd in source_wrds.sentences[i]:
                s_total.setdefault(e_wrd, 0)
                for f_wrd in target_wrds.sentences[i]:
                    s_total[e_wrd] += all_tuples[(e_wrd, f_wrd)]

            ### Collect counts
            for e_wrd in source_wrds.sentences[i]:
                for f_wrd in target_wrds.sentences[i]:
                    count[(e_wrd, f_wrd)] += all_tuples[(e_wrd, f_wrd)]/float(s_total[e_wrd])
                    total[f_wrd] += all_tuples[(e_wrd, f_wrd)]/float(s_total[e_wrd])

        ### Estimate Probabilities
        print "Estimating Probabilities"
        for e_wrd, f_wrd in all_tuples:
            all_tuples[(e_wrd, f_wrd)] = count[(e_wrd, f_wrd)]/(float)(total[f_wrd])
                
    return all_tuples

def get_alignments(source_wrds, target_wrds, all_tuples, alignment_file):
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
        
    with open(alignment_file, "w") as out:
        for i in range(len(alignments)):
            out.write(alignments[i] + "\n")
        
    return alignments


#Get the training files
train_de_path = sys.argv[1]
train_en_path = sys.argv[2]
alignment_file = sys.argv[3]
tuple_file = alignment_file.split(".txt")[0] + '.pkl'

source_wrds = Corpus(train_en_path)
target_wrds = Corpus(train_de_path)
all_tuples, foreign = create_tuples(source_wrds, target_wrds)
print "Done creating Tuples"
no_epochs = 15
all_tuples = all_tuples.fromkeys(all_tuples.keys(), 1/float(len(target_wrds.vocab.keys())))
print "Done with uniform dist"
all_tuples = create_align(no_epochs, all_tuples, foreign, source_wrds, target_wrds)
print "Done with create align"
#alignments = get_alignments(source_wrds, target_wrds, all_tuples, alignment_file)
print "Done with write tuple"
no_epochs_ibm = 1
mod_tuples = run_ibm2(source_wrds, target_wrds, all_tuples, no_epochs_ibm)
with open(tuple_file, "w") as out:
    pickle.dump(mod_tuples, out)
ibm2_alignments = get_alignments(source_wrds, target_wrds, mod_tuples, alignment_file)

