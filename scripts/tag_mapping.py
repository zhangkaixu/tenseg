#!/usr/bin/env python3
from __future__ import print_function
import sys
from collections import Counter
tagmap = """AD  ADV
AS  PRT
BA  X
CC  CONJ
CD  NUM
CS  CONJ
DEC PRT
DEG PRT
DER PRT
DEV PRT
DT  DET
ETC PRT
FW  X
IJ  X
JJ  ADJ
LB  X
LC  PRT
M   NUM
MSP PRT
NN  NOUN
NR  NOUN
NT  NOUN
OD  NUM
ON  X
P   ADP
PN  PRON
PU  .
SB  X
SP  PRT
VA  VERB
VC  VERB
VE  VERB
VV  VERB
X   X"""

def get_tm():
    tm = {}
    for line in tagmap.splitlines():
        k, v = line.split()
        tm[k] = v
    return tm

def inv():
    tm = get_tm()
    postags = {}
    for line in sys.stdin :
        line = line.split()
        line = [item.split('_') for item in line]
        line = [item for item in line if len(item) == 2]
        for w, t in line:
            if t not in postags :
                postags[t] = Counter()
            postags[t].update({w:1})
    for tag, d in postags.items():
        print(tag, tm.get(tag, "NONE"), sum(d.values()))
        ex = ' '.join(k for k, v in d.most_common(10))
        print(ex)

if __name__ == '__main__':
    tm = get_tm()
    for line in sys.stdin :
        line = line.split()
        line = [item.split('_') for item in line]
        line = [w + '_' + tm.get(t, 'X') for w, t in line]
        line = ' '.join(line)
        print(line)

