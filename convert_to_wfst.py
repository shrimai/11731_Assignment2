from collections import defaultdict
import sys

states = {0:defaultdict(lambda: len(states))}
prev = 0
nex = 0

def read_file(filename):
    with open(filename, "r") as inp:
        data = inp.read()
        
    data = data.split("\n")
    data = filter(None, data)
    return data

phrase_file = sys.argv[1]
wfst_file = sys.argv[2]
all_phrases = read_file(phrase_file)

with open(wfst_file, "w") as outf:
    for ph in all_phrases:
        tmp = ph.split("\t")
        source = tmp[1].split()
        target = tmp[0].split()
        score = tmp[2]

        for wrd in source:
            out = wrd + '<eps>'
            nex = states[prev][out]
            if nex not in states:
                states[nex] = defaultdict(lambda: len(states))
                outf.write(str(prev) + ' ' + str(nex) + ' ' + str(wrd) + ' ' + '<eps>' + '\n')
            prev = nex            
        for wrd in target:
            out = '<eps>' + wrd
            nex = states[prev][out]            
            if nex not in states:
                states[nex] = defaultdict(lambda: len(states))
                outf.write(str(prev) + ' ' + str(nex) + ' ' + '<eps>' + ' ' + str(wrd) + '\n')                
            prev = nex
        outf.write(str(prev) + ' ' + '0' + ' ' + '<eps>' + ' ' + '<eps>' + ' ' + str(score) + '\n')
        prev = 0
     
    outf.write('0 0 </s> </s>' + '\n')
    outf.write('0 0 <unk> <unk>' + '\n')
    outf.write('0')
        
 
