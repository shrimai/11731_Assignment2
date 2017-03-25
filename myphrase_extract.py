from collections import Counter
import codecs
import sys
import itertools
from collections import defaultdict
import math
import pickle

max_len = 2

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
        

def get_alignment(source_wrds, target_wrds, all_tuples):
    bwd_align = {}
    fwd_align = defaultdict(list)
    for f, f_wrd in enumerate(target_wrds):
        tmp = [(x, f_wrd) for x in source_wrds]
        prob = [all_tuples[x] for x in tmp]
        index = prob.index(max(prob))
        fwd_align[index].append(f)
        bwd_align[f] = index
    #print fwd_align  
    return fwd_align, bwd_align
    
def quasi_check(tp, bwd_align):
    l = len(tp)
    if l == 1:
        return True
    if l > max_len:
        return False
    
    tp = sorted(tp)
    actual_range = range(tp[0], tp[-1]+1)
            
    check = list(set(actual_range) - set(tp))
    for i in check:
        if bwd_align.has_key(i):
            return False
    return True
    
def check_subset(i1, i2, source_phrase):
    for s in source_phrase:
        if s < i1 or s > i2:
            return False
    return True
    
def extract_phrases(source_sent, target_sent, all_tuples):
    extracted_phrases = []
    fwd_align, bwd_align = get_alignment(source_sent, target_sent, all_tuples)
    for i1 in range(len(source_sent)):
        target_phrases = []
        for i2 in range(i1, len(source_sent)):
            if i2 - i1 > max_len:
                continue
            
            for i in range(i1, i2+1):
                target_phrases.extend(fwd_align[i])
            target_phrases = list(set(target_phrases))
            
            if len(target_phrases) != 0 and quasi_check(target_phrases, bwd_align):
                j1 = min(target_phrases)
                j2 = max(target_phrases)                
                
                if j2 - j1 > max_len:
                    continue

                source_phrase = []
                for j in range(j1, j2+1):
                    source_phrase.append(bwd_align[j])
                    
                if check_subset(i1, i2, source_phrase) and len(source_phrase) != 0:
                    e_phrase = source_sent[i1:i2+1]
                    f_phrase = target_sent[j1:j2+1]
                    extracted_phrases.append((' '.join(e_phrase), ' '.join(f_phrase)))
                    #print extracted_phrases
                
                    while j1 >= 0 and not bwd_align.has_key(j1):
                        j_prime = j2
                        
                        if j_prime - j1 > max_len:
                            continue
                            
                        while j_prime < len(f) and not bwd_align.has_key(j2): 
                            f_phrase = f[j1:j_prime+1]
                            extracted_phrases.append((' '.join(e_phrase), ' '.join(f_phrase)))
                            j_prime += 1
                        j1 -= 1
                    
    return extracted_phrases  
    
def calculate_score(phrases):
    
    for (source, target) in phrases:
        count_sgivent[(source, target)] += 1.0
        count_s[source] += 1.0
        score_sgivent[(source, target)] = math.log(count_sgivent[(source, target)]/float(count_s[source]))
        if score_sgivent[(source, target)] != 0.0 :
            score_sgivent[(source, target)] = -score_sgivent[(source, target)]

train_de_path = sys.argv[1]
train_en_path = sys.argv[2]
alignment_file = sys.argv[3]
phrase_file = sys.argv[4]
tuple_file = alignment_file.split(".txt")[0] + '.pkl'

source_wrds = Corpus(train_en_path)
target_wrds = Corpus(train_de_path)
with open(tuple_file, "r") as inp:
    all_tuples = pickle.load(inp)
    
count_sgivent = defaultdict(int)
count_s = defaultdict(int)
score_sgivent = defaultdict(float)
final_phrases = []
print "Starting phrase extract"
for i in range(len(source_wrds.sentences)):

    phrases = extract_phrases(source_wrds.sentences[i], target_wrds.sentences[i], all_tuples)
    calculate_score(phrases)
    final_phrases.extend(phrases)
    
final_phrases = list(set(final_phrases))
print "Writing phrase extract"
with open(phrase_file, "w") as out:
    for src_phrase, tgt_phrase in final_phrases:
            out.write(src_phrase + '\t' + tgt_phrase + '\t')
            out.write(str(score_sgivent[(src_phrase, tgt_phrase)]))
            out.write('\n')   
