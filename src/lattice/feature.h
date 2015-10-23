#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "common/common.h"
#include "common/weight.h"
#include "common/dictionary.h"

namespace tenseg {

using namespace std;
//class Feature;
//
template<class SPAN>
class LabelledFeature {
public:
    LabelledFeature() :_transition_ptr(nullptr),
        _dict(nullptr) {};

    void set_tag_indexer(shared_ptr<Indexer<string>> tag_indexer) {
        _tag_indexer = tag_indexer;
    }
    void set_weight(Weight& dict) {
        _dict = &dict;
    }
    void set_dictionary(shared_ptr<Dictionary> dictionary) {
        _dictionary = dictionary;
    }

    void prepare(
            const string& raw,
            const vector<size_t>& off,
            const vector<SPAN>& lattice) {
        if (_dict == nullptr) {
            fprintf(stderr, "no weight are set for feature");
            return;
        }
        _lattice = &lattice;
        _off = &off;
        _raw = &raw;
        _transition_ptr = _dict->get("transition");
        _calc_emission(*_dict, *_raw, _emission, false);

        _labels.clear();
        _label_index.clear();
        for (size_t i = 0; i < lattice.size(); i++) {
            const SPAN& span = lattice[i];
            _label_index.push_back(_tag_indexer->get(span.label()));
            _labels.push_back(
                        _tag_indexer->get(span.label()) * 2
                        + ((span.end - span.begin == 1)?0:1)
                    );
        }
    }

    void calc_gradient(
            vector<SPAN>& gold, 
            vector<SPAN>& output, 
            Weight& gradient) {

        /// is eaual
        if (gold.size() == output.size()) {
            bool is_equal = true;
            for (size_t i = 0; i < gold.size(); i++) {
                if (!(gold[i] == output[i])) {
                    is_equal = false; break;
                }
            }
            if (is_equal) { return; }
        }

        /// character based
        _emission.clear();
        _emission.insert(_emission.end(), (N * tagset_size() * gold.back().end), 0);
        for (size_t i = 0; i < gold.size(); i++) {
            _update_span_emi(gold[i], 1);
        }
        for (size_t i = 0; i < output.size(); i++) {
            _update_span_emi(output[i], -1);
        }
        _calc_emission(gradient, *_raw, _emission, true);

        /// dictionary based
        if (_dictionary) {
            for (size_t i = 0; i < gold.size(); i++) {
                _unigram_dictionary_gradient(&gold[i], gradient, 1);
            }
            for (size_t i = 0; i < output.size(); i++) {
                _unigram_dictionary_gradient(&output[i], gradient, -1);
            }
        }

        /// bigram
        vector<double> g_trans(2 * _tag_indexer->size() * 2 * _tag_indexer->size());
        _update_g_trans(g_trans, gold, 1);
        _update_g_trans(g_trans, output, -1);
        gradient.add_from("transition", &g_trans[0], g_trans.size());
    }


    /**
     * interface to calc unigram scores
     * */
    double unigram(size_t uni) {
        const SPAN& span = (*_lattice)[uni];
        double score = 0;

        size_t l = _label_index[uni];

        /// char-based features
        if (span.end - span.begin == 1) { // S
            score += _emission[span.begin * N * tagset_size() + N * l + 3];
        } else { // BM*E
            score += _emission[span.begin * N * tagset_size() + N * l + 0];
            for (size_t i = span.begin + 1; i < span.end - 1; i++) {
                score += _emission[i * N * tagset_size() + N * l + 1];
            }
            score += _emission[(span.end - 1) * N * tagset_size() + N * l + 2];
        }

        /// word-based features
        
        /// dict-based
        score += unigram_dictionary(uni, &span);
        return score;
    }

    /**
     * interface to calc bigram scores
     * */
    inline double bigram(size_t first, size_t second) {
        double* ptr = _transition_ptr;
        if (ptr == nullptr) return 0;
        return ptr[_trans_ind(first, second)];
    }

    /**
     * 计算字ngram特征
     * */
    void _calc_char_ngram_emision(
            const size_t n,
            const string& raw,
            const vector<size_t>& begins,
            Weight& model,
            vector<double>& emission, bool update
            ) {
        for (size_t i = 0; i < begins.size() - n; i++) {
            string uni = raw.substr(begins[i], 
                    begins[i + n] - begins[i]);

            if (n == 1 && uni[0] == '|') {
                uni = string("，");
            }

            double* m = model.get(uni);
            if (m == nullptr) {
                if (update == false) {
                    continue;
                } else {
                    model.insert(uni, (n + 2) * N * tagset_size());
                    m = model.get(uni);
                }
            };


            int b = (((int)i - 1) * (int)N * (int)tagset_size());
            int e = (min(((int)(2 + n)), ((int)begins.size() - (int)i))
                    * N * tagset_size());
            int j = max(0, - b);
            double *eo = emission.data() + b;

            if (update == false) {
                for (; j < e; j++) {
                    eo[j] += m[j];
                }
            } else {
                for (; j < e; j++) {
                    m[j] += eo[j];
                }
            }
        }
    }

    void _calc_emission(Weight& model, const string& raw,
            vector<double>& emission, bool update) {
        if (!update) {
            emission.clear();
            emission.insert(emission.end(), 
                    N * tagset_size() * (_off->size() - 1), 0);
        }
        _calc_char_ngram_emision(1, raw, *_off, model, emission, update);
        _calc_char_ngram_emision(2, raw, *_off, model, emission, update);
    }

    void _update_span_emi(SPAN& span, double delta) {
        size_t l = _tag_indexer->get(span.label());
        if (span.end - span.begin == 1) {
            _emission[span.begin * N * tagset_size() + N * l + 3] += delta;
        } else {
            _emission[span.begin * N * tagset_size() + N * l + 0] += delta;
            for (size_t i = span.begin + 1; i < span.end - 1; i++) {
                _emission[i * N * tagset_size() + N * l + 1] += delta;
            }
            _emission[(span.end - 1) * N * tagset_size() + N * l + 2] += delta;
        }
    }
    double _unigram_dictionary_gradient(const SPAN* span, Weight& gradient, double delta) {
        if (!_dictionary) return 0;
        string key = _raw->substr((*_off)[span->begin], (*_off)[span->end] - (*_off)[span->begin]);
        if (!_dictionary->get(key, key)) return 0;
        key = string("dic:") + key;

        //vector<double> g(_tag_indexer->size() + 1);
        //g[0] += delta;
        //g[_tag_indexer->get(span->label()) + 1] += delta;
        //gradient.add_from(key, g.data(), g.size());
        gradient.add_from(key, &delta, 1);
    }

    double unigram_dictionary(size_t ind, const SPAN* span) {
        if (!_dictionary) return 0;

        string key = _raw->substr((*_off)[span->begin], (*_off)[span->end] - (*_off)[span->begin]);
        
        if (!_dictionary->get(key, key)) return 0;

        key = string("dic:") + key;
        double* value = _dict->get(key);

        if (!value) return 0;
        //return value[0] + value[_label_index[ind] + 1];
        return *value;
    }
    inline size_t _trans_ind(size_t a, size_t b){
        return _labels[a] * 2 * _tag_indexer->size() + _labels[b];
    }

    inline size_t _trans_ind(const SPAN& span_a, const SPAN& span_b){
        size_t i_a =_tag_indexer->get(span_a.label()) * 2 
            + ((span_a.end - span_a.begin == 1)?0:1);
        size_t i_b =_tag_indexer->get(span_b.label()) * 2 
            + ((span_b.end - span_b.begin == 1)?0:1);
        return i_a * 2 * _tag_indexer->size() + i_b;
    }

    void _update_g_trans(vector<double>& g_trans, vector<SPAN>& seq, double delta) {
        for (size_t i = 0; i < seq.size() - 1; i++) {
            SPAN& span_a = seq[i];
            SPAN& span_b = seq[i + 1];

            g_trans[_trans_ind(span_a, span_b)] += delta;
        }
    }
    size_t tagset_size() const {
        if (_tag_indexer) {
            return _tag_indexer->size();
        }
        return 1;
    }
private:
    const size_t N = 4;

    const string* _raw;
    const vector<size_t>* _off;
    const vector<SPAN>* _lattice;


    /// features
    Weight* _dict;
    /// char-based related
    double* _transition_ptr;
    vector<size_t> _labels;
    vector<size_t> _label_index;
    vector<double> _emission;

    shared_ptr<Indexer<string>> _tag_indexer;
    
    shared_ptr<Dictionary> _dictionary;


};

}
