all : char_segger word_segger

char_segger : char_segger.cc char_dict.h char_searcher.h char_eval.h
	g++ --std=c++11 -g -O3 $< -o $@

word_segger : word_based.cc tenseg.h feature.h weight.h optimizer.h
	g++ --std=c++11 -g -O3 $< -o $@
