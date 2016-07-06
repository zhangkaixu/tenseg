#!/usr/bin/env python3
import sys


def to_set(words):
    off = 0
    s = set()
    for w in words:
        s.add((off, off + len(w)))
        off += len(w)
    return s

class Overlap :
    def __init__(self, spans):
        begin = {}
        end = {}
        for b, e in spans:
            if b not in begin or begin[b] < e:
                begin[b] = e
            if e not in end or end[e] > b:
                end[e] = b
        self._begin = begin
        self._end = end
    def __call__(self, span) :
        b = span[0]
        e = span[1]
        for i in range(b + 1, e):
            if i in self._begin and self._begin[i] > e :
                return False
            if i in self._end and self._end[i] < b:
                return False
        return True



def check(std, rst):
    std_set = to_set(std)

    over = Overlap(std_set)
    
    off = 0
    out = []
    for w in rst :
        span = (off, off + len(w))
        ow = w
        if not over(span):
            ow = "\033[41m%s"%(ow)
        elif span not in std_set:
            ow = "\033[5m%s"%(ow)

        off += len(w)
        ow += '\033[0m'
        out.append(ow)
    print(' '.join(out))

    
    pass
if __name__ == '__main__':
    if len(sys.argv) < 3 :
        exit()
    for std, rst in zip(open(sys.argv[1]), open(sys.argv[2])):
        std = std.split()
        rst = rst.split()
        print(*std)
        check(std, rst)
    pass
