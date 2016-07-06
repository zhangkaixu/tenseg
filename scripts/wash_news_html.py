#!/usr/bin/env python2
#coding:utf8
from __future__ import print_function
import sys
import gzip
import re
import logging
import json
from readability.readability import Document

#logging.basicConfig(level=logging.ERROR)
"""
对输入流的html进行清洗
html的分割为`<docurl>XX</docurl>`
输出json格式结果
"""

def decode_doc(doc, url):
    #print('doc')
    cs = re.compile(b'^<(meta|META).*charset=("|\')?([^ "\']*)')
    pkey = re.compile(b'^<(meta|META).*keywords.*content=("|\')?([^ "\']*)')
    codec = None
    keywords = None
    #print(*doc)
    for l in doc :
        if (l.startswith(b'<meta') or l.startswith(b'<META')) :
            if codec is None and (b'charset' in l) :
                m = cs.match(l)
                codec = m.group(3).decode()
            if keywords is None and b'keywords' in l :
                m = pkey.match(l)
                if m :
                    keywords = m.group(3)


    sdoc = []
    for l in doc :
        try :
            l = l.decode(codec)
        except :
            l = ''
        sdoc.append(l)

    try :
        if keywords :
            keywords = keywords.decode(codec)
        else :
            #print(*sdoc, sep = '\n')
            keywords = ''
        keywords = re.split(r'[ ,;\|]',keywords)
        #print(keywords.encode('utf8'))
    except :
        pass

    #if sum(len(x) for x in sdoc) < 1000 : return
    doc = '\n'.join(sdoc)
    #if len(doc) < 1000 :return
    try :
        doc = Document(doc)
        title = doc.short_title()
        content = doc.summary()
    except :
        return
    #print(doc.summary().encode('utf8'))
    #print(doc.short_title().encode('utf8'))


    data = {"url":url, 
            'keywords':keywords,
            'title': title,
            'content':content}
    return data


    #features = []
    #punc = set('。，？！：；“”')
    #for l in sdoc :
    #    total = len(l)
    #    p = len(re.findall('<p', l))
    #    chinese = sum(1 for c in l if c >= u'一' and c <= unichr(40866))
    #    puncs = sum(1 for c in l if c in punc)
    #    print(total, chinese, puncs, p)

    #    pass
    ##print(sdoc)
    #pass

def gen_doc():
    cache = []
    url = ''
    n = 0
    for line in sys.stdin:
        if line.startswith(b'<docurl>'):
            n = n + 1
            print(n, file = sys.stderr, end = '\r')
            if cache :
                data = decode_doc(cache, url)
                yield data
            url = line.partition('>')[-1].partition('<')[0]
            cache = []
        else :
            cache.append(line.strip())
    if cache :
        data = decode_doc(cache, url)
        yield data

if __name__ == '__main__':
    for doc in gen_doc():
        if doc is None : 
            continue
        try :
            s = json.dumps(doc, ensure_ascii = False).encode('utf8')
            print(s)
            sys.stdout.flush()
        except:
            pass
