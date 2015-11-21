#pragma once
#include "lattice/feature.h"

#include "common/common.h"
#include "common/weight.h"
#include "common/dictionary.h"

#include <cmath>

namespace tenseg {

template<class SPAN>
class UnigramFeature : public ILatticeFeature<SPAN> {
public:
    UnigramFeature(const string& filename) {
        auto dictionary = make_shared<Dictionary<double>>();
        dictionary->load(filename.c_str());
        _dict = dictionary;
        _weight_prefix = "d:" + filename + ":";
        _bigram_weight_prefix = "d:" + filename + ":b:";
    }
    virtual void prepare(shared_ptr<string>& raw, shared_ptr<vector<size_t>>& off, vector<SPAN>& lattice) {
        _lattice = &lattice;
        _raw = raw;
        _off = off;
    }
    virtual double unigram(size_t ind) {
        if (!_dict) return 0;
        double score = _uni_freq((*_lattice)[ind]);
        return score - 8;
    }

    virtual void calc_gradient(vector<SPAN>& gold, vector<SPAN>& output, Weight& gradient) {
        //for (size_t i = 0; i < gold.size(); i++) {
        //    _unigram_gradient(gold[i], gradient, 1);
        //}
        //for (size_t i = 0; i < output.size(); i++) {
        //    _unigram_gradient(output[i], gradient, -1);
        //}
    }

private:
    /**
     * brief : calc unigram key
     * */
    double _uni_freq(const SPAN& span) {
        string unigram_string = _raw->substr((*_off)[span.begin], (*_off)[span.end] - (*_off)[span.begin]);
        double freq = 0;
        if (!_dict->get(unigram_string, freq)) {
            freq = 0;
        }  else {
        }
        freq = log10(freq + 1);
        return freq;
    }

    //double _unigram_gradient(const SPAN& span, Weight& gradient, double delta) {
    //    if (!_dict) return 0;
    //    string key;
    //    if (!_uni_key(span, key)) return 0;
    //    gradient.add_from(key, &delta, 1);
    //}
private:
    string _weight_prefix;
    string _bigram_weight_prefix;
    shared_ptr<Dictionary<double>> _dict;

    vector<SPAN>* _lattice;
    shared_ptr<string> _raw;
    shared_ptr<vector<size_t>> _off;
};

}
