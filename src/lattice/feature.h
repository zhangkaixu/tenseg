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
class ILatticeFeature {
public:
    ILatticeFeature(){};
    virtual void prepare(shared_ptr<string>& raw, shared_ptr<vector<size_t>>& off, vector<SPAN>& lattice) {}
    virtual double unigram(size_t uni) {return 0;}
    virtual double bigram(size_t first, size_t second) {return 0;}
    virtual void calc_gradient(vector<SPAN>& gold, vector<SPAN>& output, Weight& gradient) {}
    void set_weight(Weight& weight) { _weight = &weight; }
protected:
    Weight* _weight;
};


template<class SPAN>
class DictFeature : public ILatticeFeature<SPAN> {
public:
    DictFeature(const string& filename) {
        auto dictionary = make_shared<Dictionary>();
        dictionary->load(filename.c_str());
        _dict = dictionary;
        _weight_prefix = "d:" + filename + ":";
    }
    
    DictFeature(shared_ptr<Dictionary>& dict) {
        _dict = dict;
        _weight_prefix = "dic:";
    }
    
    virtual void prepare(shared_ptr<string>& raw, shared_ptr<vector<size_t>>& off, vector<SPAN>& lattice) {
        _lattice = &lattice;
        _raw = raw;
        _off = off;
    }
    virtual double unigram(size_t ind) {
        const SPAN span = (*_lattice)[ind];

        if (!_dict) return 0;

        string key = _raw->substr((*_off)[span.begin], (*_off)[span.end] - (*_off)[span.begin]);
        
        if (!_dict->get(key, key)) return 0;

        key = _weight_prefix + key;
        double* value = this->_weight->get(key);

        if (!value) return 0;
        return *value;
    }
    virtual void calc_gradient( vector<SPAN>& gold, vector<SPAN>& output, Weight& gradient) {
        for (size_t i = 0; i < gold.size(); i++) {
            _unigram_gradient(&gold[i], gradient, 1);
        }
        for (size_t i = 0; i < output.size(); i++) {
            _unigram_gradient(&output[i], gradient, -1);
        }
    }

private:
    double _unigram_gradient(const SPAN* span, Weight& gradient, double delta) {
        if (!_dict) return 0;
        string key = _raw->substr((*_off)[span->begin], (*_off)[span->end] - (*_off)[span->begin]);
        if (!_dict->get(key, key)) return 0;
        key = _weight_prefix + key;
        gradient.add_from(key, &delta, 1);
    }
private:
    string _weight_prefix;
    shared_ptr<Dictionary> _dict;

    vector<SPAN>* _lattice;
    shared_ptr<string> _raw;
    shared_ptr<vector<size_t>> _off;
};

template<class SPAN>
class PhraseFeature : public ILatticeFeature<SPAN> {
public:
    PhraseFeature(const string& filename) {
        auto dictionary = make_shared<Dictionary>();
        dictionary->load(filename.c_str());
        _phrase = dictionary;
        _weight_prefix = "p:" + filename + ":";
    }
    virtual void prepare(shared_ptr<string>& raw, shared_ptr<vector<size_t>>& off, vector<SPAN>& lattice) {
        _lattice = &lattice;
        _raw = raw;
        _off = off;

        while (_span_extra_info.size() < lattice.size()) {
            _span_extra_info.push_back(span_extra_info_t());
        }

        for (size_t i = 0; i < lattice.size(); i++) {
            _span_extra_info[i].reset();
        }
        _prepare_phrase();
    }

    double unigram(size_t ind) {
        if (!_phrase) return 0;
        if (_span_extra_info[ind].phrase_conflict_ == 0) return 0;
        string key = _weight_prefix;
        double* value = this->_weight->get(key);
        if (!value) return 0;
        return *value;
    }
    virtual void calc_gradient( vector<SPAN>& gold, vector<SPAN>& output, Weight& gradient) {
        for (size_t i = 0; i < gold.size(); i++) {
            _unigram_phrase_gradient(&gold[i], gradient, 1);
        }
        for (size_t i = 0; i < output.size(); i++) {
            _unigram_phrase_gradient(&output[i], gradient, -1);
        }
    }


private:
    void _prepare_phrase() {
        if (!_phrase) return;
        _phrase_list.clear();
        vector<SPAN> _tmp_phrase_list;

        const size_t MAX_PHRASE = 12;
        while (_phrase_begins.size() < _off->size()) {
            _phrase_begins.push_back(vector<size_t>());
        }
        while (_phrase_ends.size() < _off->size()) {
            _phrase_ends.push_back(vector<size_t>());
        }
        for (size_t i = 0; i < _off->size(); i++) {
            _phrase_begins[i].clear();
            _phrase_ends[i].clear();
        }

        /// 找到所有phrase
        for (size_t i = 0; i < _off->size() - 1; i ++) {
            size_t begin = (*_off)[i];
            for (size_t j = i + 1; j < i + MAX_PHRASE; j ++) {
                if (j >= _off->size()) break;
                size_t end = (*_off)[j];
                string value;
                if (_phrase->get(_raw->substr(begin, end - begin), value)) {
                    _tmp_phrase_list.push_back(SPAN(i, j));
                    //printf("phrase %s %lu %lu\n",_raw->substr(begin, end - begin).c_str(), i, j);
                    //_phrase_begins[i].push_back(j);
                    //_phrase_ends[j].push_back(i);
                }
            }
        }
        /// 过滤掉overlap的phrase
        for (auto& pa : _tmp_phrase_list) {
            bool ok = true;
            size_t a = pa.begin;
            size_t b = pa.end;
            for (auto& pb : _tmp_phrase_list) {
                if (!ok) break;
                size_t c = pb.begin;
                size_t d = pb.end;
                if ((a < c && c < b && b < d) ||
                        (c< a && a < d && d < b)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                _phrase_list.push_back(pa);
            }
        }

        /// 填写begin end
        for (auto& span : _phrase_list) {
            size_t i = span.begin;
            size_t j = span.end;
            _phrase_begins[i].push_back(j);
            _phrase_ends[j].push_back(i);
        }

        /// 为lattice计算是否冲突
        for (size_t i = 0; i < _lattice->size(); i++) {
            auto& span = (*_lattice)[i];
            for (size_t j = span.begin + 1; j < span.end; j++) {
                if (_span_extra_info[i].phrase_conflict_ == 1) break;
                for (auto phrase_begin : _phrase_ends[j]) {
                    if (phrase_begin < span.begin) {
                        _span_extra_info[i].phrase_conflict_ = 1;
                        break;
                    }
                }
                for (auto phrase_end : _phrase_begins[j]) {
                    if (phrase_end > span.end) {
                        _span_extra_info[i].phrase_conflict_ = 1;
                        break;
                    }
                }
            }
            if (_span_extra_info[i].phrase_conflict_) {
                size_t b = (*_off)[span.begin];
                size_t e = (*_off)[span.end];
            }
        }
    }
    double _unigram_phrase_gradient(const SPAN* span, Weight& gradient, double delta) {
        if (!_phrase) return 0;

        bool conflict = false;

        for (size_t j = span->begin + 1; j < span->end; j++) {
            if (conflict) break;
            for (auto phrase_begin : _phrase_ends[j]) {
                if (phrase_begin < span->begin) {
                    conflict = true;
                    break;
                }
            }
            for (auto phrase_end : _phrase_begins[j]) {
                if (phrase_end > span->end) {
                    conflict = true;
                    break;
                }
            }
        }
        if (conflict) {
            string key = _weight_prefix;
            gradient.add_from(key, &delta, 1);
        }
    }

    struct span_extra_info_t {
        size_t phrase_conflict_;
        void reset() {
            phrase_conflict_ = 0;
        }
    };
    vector<span_extra_info_t> _span_extra_info;

    string _weight_prefix;
    shared_ptr<Dictionary> _phrase;

    vector<SPAN> _phrase_list;
    vector<vector<size_t>> _phrase_begins;
    vector<vector<size_t>> _phrase_ends;

    vector<SPAN>* _lattice;
    shared_ptr<string> _raw;
    shared_ptr<vector<size_t>> _off;
};


template<class SPAN>
class LabelledFeature {
private:
    struct span_extra_info_t {
        size_t phrase_conflict_;
        void reset() {
            phrase_conflict_ = 0;
        }
    };
public:
    LabelledFeature() :_transition_ptr(nullptr),
        _dict(nullptr) {};

    void set_tag_indexer(shared_ptr<Indexer<string>> tag_indexer) {
        _tag_indexer = tag_indexer;
    }
    void set_weight(Weight& dict) {
        _dict = &dict;
        for (auto& f : _features) {
            f->set_weight(dict);
        }
    }
    vector<shared_ptr<ILatticeFeature<SPAN>>>& features() {
        return _features;
    }
    

    void prepare(
            shared_ptr<string>& raw,
            shared_ptr<vector<size_t>>& off,
            vector<SPAN>& lattice) {
        if (_dict == nullptr) {
            fprintf(stderr, "no weight are set for feature");
            return;
        }
        for (auto& f : _features) {
            //printf("pre\n");
            f->prepare(raw, off, lattice);
        }
        _lattice = &lattice;
        to_half(*raw, *off, _raw, _off);
        _transition_ptr = _dict->get("transition");
        _calc_emission(*_dict, _raw, _emission, false);

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
        for (auto& f : _features) {
            f->calc_gradient(gold, output, gradient);
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
        _calc_emission(gradient, _raw, _emission, true);


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
        double score = 0;
        for (auto& f : _features) {
            score += f->unigram(uni);
        }
        const SPAN& span = (*_lattice)[uni];

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
        return score;
    }

    /**
     * interface to calc bigram scores
     * */
    inline double bigram(size_t first, size_t second) {
        double score = 0;
        for (auto& f : _features) {
            score += f->bigram(first, second);
        }
        double* ptr = _transition_ptr;
        if (ptr == nullptr){
        } else {
            score += ptr[_trans_ind(first, second)];
        };
        return score;
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
                    N * tagset_size() * (_off.size() - 1), 0);
        }
        _calc_char_ngram_emision(1, raw, _off, model, emission, update);
        _calc_char_ngram_emision(2, raw, _off, model, emission, update);
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

    string _raw;
    vector<size_t> _off;
    const vector<SPAN>* _lattice;


    /// features
    Weight* _dict;
    /// char-based related
    double* _transition_ptr;
    vector<size_t> _labels;
    vector<size_t> _label_index;
    vector<double> _emission;

    shared_ptr<Indexer<string>> _tag_indexer;
    
    vector<shared_ptr<ILatticeFeature<SPAN>>> _features;


};

}
