#!/usr/bin/env python3
import sys
import random


if __name__ == '__main__':
    data = [line.rstrip() for line in sys.stdin]
    random.shuffle(data)
    for line in data:
        print(line)
