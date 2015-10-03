#pragma
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "tenseg.h"
#include "weight.h"
#include "dictionary.h"

namespace tenseg {

using namespace std;
class Feature;

class Feature {
    const size_t N = 4;

    const string* _raw;
    const vector<size_t>* _off;
    const vector<linked_span_t>* _lattice;

    /// features
    Weight* _dict;
    /// char-based related
    vector<double> _transition;
    vector<double> _emission;
    
    shared_ptr<Dictionary> _dictionary;

    void _calc_emission(Weight& model, const string& raw,
            vector<double>& emission, bool update) {
        vector<size_t> begins;
        for (size_t i = 0; i < raw.size(); i++) {
            if ((0xc0 == (raw[i] & 0xc0))
                    || !(raw[i] & 0x80)) { // first char
                begins.push_back(i);
            }
        }
        begins.push_back(raw.size());

        if (!update) {
            emission.clear();
            for (size_t i = 0; i < N * (begins.size() - 1); i ++) {
                emission.push_back(0);
            }
        }

        int n = 1;
        for (size_t i = 0; i < begins.size() - n; i++) {
            string uni = raw.substr(begins[i], 
                    begins[i + n] - begins[i]);

            if (uni[0] == '|') {
                uni = string("，");
            }

            double* m = model.get(uni);
            if (m == nullptr) {
                if (update == false) {
                    continue;
                } else {
                    model.insert(uni, (n + 2) * N);
                    m = model.get(uni);
                }
            };

            for (int j = 0; j < (2 + n) * N; j++) {
                int off = ((int)i - 1) * N + j;
                if (off < 0) continue;
                if (off >= (begins.size() - 1) * N) continue;
                if (update == false) {
                    emission[off] += m[j];
                } else {
                    m[j] += emission[off];
                }
            }
        }

        n = 2;
        if (begins.size() > n) {
            for (size_t i = 0; i < (int)(begins.size()) - n; i++) {
                string uni = raw.substr(begins[i], 
                        begins[i + n] - begins[i]);

                double* m = model.get(uni);
                if (m == nullptr) {
                    if (update == false) {
                        continue;
                    } else {
                        model.insert(uni, (n + 2) * N);
                        m = model.get(uni);
                    }
                };

                for (int j = 0; j < (2 + n) * N; j++) {
                    int off = ((int)i - 1) * N + j;
                    if (off < 0) continue;
                    if (off >= (begins.size() - 1) * N) continue;
                    if (update == false) {
                        emission[off] += m[j];
                    } else {
                        m[j] += emission[off];
                    }
                }
            }
        }
    }
    void _update_span_emi(span_t& span, double delta) {
        if (span.end - span.begin == 1) {
            _emission[span.begin * N + 3] += delta;
        } else {
            _emission[span.begin * N + 0] += delta;
            for (size_t i = span.begin + 1; i < span.end - 1; i++) {
                _emission[i * N + 1] += delta;
            }
            _emission[(span.end - 1) * N + 2] += delta;
        }
    }
public:
    Feature() {};

    void set_dict(Weight& dict) {
        _dict = &dict;
    }
    void set_dictionary(shared_ptr<Dictionary> dictionary) {
        _dictionary = dictionary;
    }

    void prepare(
            const string& raw,
            const vector<size_t>& off,
            const vector<linked_span_t>& lattice) {
        _lattice = &lattice;
        _off = &off;
        _raw = &raw;
        _calc_emission(*_dict, *_raw, _emission, false);
    }


    double _unigram_dictionary_gradient(const span_t* span, Weight& gradient, double delta) {
        if (!_dictionary) return 0;
        string key = _raw->substr((*_off)[span->begin], (*_off)[span->end] - (*_off)[span->begin]);
        if (!_dictionary->get(key, key)) return 0;
        key = string("dic:") + key;
        gradient.add_from(key, &delta, 1);
    }

    double unigram_dictionary(const span_t* span) {
        if (!_dictionary) return 0;

        string key = _raw->substr((*_off)[span->begin], (*_off)[span->end] - (*_off)[span->begin]);
        
        if (!_dictionary->get(key, key)) return 0;

        key = string("dic:") + key;
        double* value = _dict->get(key);

        if (!value) return 0;

        return *value;
    }

    void calc_gradient(
            vector<span_t>& gold, 
            vector<span_t>& output, 
            Weight& gradient) {
        /// is eaual
        if (gold.size() == output.size()) {
            bool is_equal = true;
            for (size_t i = 0; i < gold.size(); i++) {
                if (gold[i].begin != output[i].begin) {
                    is_equal = false;
                    break;
                }
            }
            if (is_equal) {
                return;
            }
        }



        /// character based
        _emission.clear();
        for (size_t i = 0; i < (N * gold.back().end); i++) {
            _emission.push_back(0);
        }
        for (size_t i = 0; i < gold.size(); i++) {
            _update_span_emi(gold[i], 1);
        }
        for (size_t i = 0; i < output.size(); i++) {
            _update_span_emi(output[i], -1);
        }

        _calc_emission(gradient, *_raw, _emission, true);

        /// dictionary based
        for (size_t i = 0; i < gold.size(); i++) {
            _unigram_dictionary_gradient(&gold[i], gradient, 1);
        }
        for (size_t i = 0; i < output.size(); i++) {
            _unigram_dictionary_gradient(&output[i], gradient, -1);
        }
    }

    /**
     * interface to calc unigram scores
     * */
    double unigram(size_t uni) {
        const linked_span_t& span = (*_lattice)[uni];
        double score = 0;

        /// char-based features
        if (span.end - span.begin == 1) { // S
            score += _emission[span.begin * N + 3];
        } else { // BM*E
            score += _emission[span.begin * N + 0];
            for (size_t i = span.begin + 1; i < span.end - 1; i++) {
                score += _emission[i * N + 1];
            }
            score += _emission[(span.end - 1) * N + 2];
        }

        /// word-based features
        
        /// dict-based
        score += unigram_dictionary(&span);

        return score;
    }

    /**
     * interface to calc bigram scores
     * */
    double bigram(size_t first, size_t second) {
        return 0;
    }
};
}
