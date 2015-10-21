#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <set>
#include <vector>
#include <memory>
#include "common/common.h"
#include "common/feature.h"
#include "common/weight.h"
#include "common/optimizer.h"
#include "common/dictionary.h"

using namespace std;

using namespace tenseg;


class Lattice_Generator {
    enum char_type_t { ///< 字符类型
        NORMAL,     ///< 普通字符
        PUNC        ///< 标点符号
    };
    set<string> _punc; ///< 标点符号集合
    vector<char_type_t> _types;
    shared_ptr<Indexer<string>> _tag_indexer;

    void _calc_type(const string& raw,
            const vector<size_t>& off) {
        _types.clear();
        for (size_t i = 0; i < off.size() - 1; i++) {
            string ch = raw.substr(off[i], off[i + 1] - off[i]);
            _types.push_back(char_type_t::NORMAL);
            if (_punc.find(ch) != _punc.end()) {
                _types.back() = char_type_t::PUNC;
            }
        }
        return;
    }
public:
    Lattice_Generator() {
        _punc.insert(string("。")); _punc.insert(string("，"));
        _punc.insert(string("？")); _punc.insert(string("！"));
        _punc.insert(string("：")); _punc.insert(string("“"));
        _punc.insert(string("”"));
    }

    void set_tag_indexer(shared_ptr<Indexer<string>> ti) {
        _tag_indexer = ti;
    }

    void gen(const string& raw, 
            const vector<size_t>& off, 
            const vector<labelled_span_t>& span,
            vector<labelled_span_t>& lattice) {

        if (off.size() == 0) return;

        _calc_type(raw, off);
        
        size_t n = off.size() - 1;

        lattice.clear();
        // generate all spans
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n + 1; j++) {
                if (j - i > 10) break;

                for (size_t k = 0; k < _tag_indexer->size(); k++) {
                    lattice.push_back(labelled_span_t(i, j, (*_tag_indexer)[k]));
                }

                if (_types[i] == char_type_t::PUNC) break;
                if (j < n && _types[j] == char_type_t::PUNC) break;
            }
        }
    }
};


class PathFinder {
private:
    vector<vector<size_t>> begins;
    vector<vector<size_t>> ends;

    vector<double> scores_;
    vector<size_t> pointers_;

public:
    PathFinder() {
    }

    template <class SPAN, class FEATURE>
    void find_path(const string& raw,
            const vector<size_t>& off,
            FEATURE& feature,
            const vector<SPAN>& lattice,
            vector<SPAN>& output
            ) {
        
        feature.prepare(raw, off, lattice);

        /// Step 1 prepare path
        while (begins.size() < off.size()) { begins.push_back(vector<size_t>()); }
        while (ends.size() < off.size()) { ends.push_back(vector<size_t>()); }
        for (size_t i = 0; i < off.size(); i++) {
            begins[i].clear(); ends[i].clear();
        }

        scores_.clear();
        pointers_.clear();
        for (size_t i = 0; i < lattice.size(); i++) {
            ends[lattice[i].end].push_back(i);
            begins[lattice[i].begin].push_back(i);

            scores_.push_back(0);
            pointers_.push_back(0);
        }

        /// Step 2 search
        for (size_t i = 0; i < off.size() - 1; i++) {
            for (size_t k = 0; k < begins[i].size(); k++) {
                double& max_score = scores_[begins[i][k]];
                size_t& max_pointer = pointers_[begins[i][k]];
                bool has_max = false;
                for (size_t j = 0; j < ends[i].size(); j ++) {
                    double score = 0;
                    size_t p = ends[i][j];
                    score = scores_[p];
                    score += feature.bigram(ends[i][j], begins[i][k]);
                    /// bigram features
                    if (!has_max || max_score < score) {
                        has_max = true;
                        max_score = score;
                        max_pointer = p;
                    }
                }
                /// unigram features
                max_score += feature.unigram(begins[i][k]);
            }
        }

        /// Step 3 find best
        double max_score = 0;
        size_t max_pointer = 0;
        bool has_max = false;
        for (size_t j = 0; j < ends[off.size() - 1].size(); j ++) {
            double score = 0;
            size_t p = ends[off.size() - 1][j];
            score = scores_[p];
            if (!has_max || max_score < score) {
                has_max = true;
                max_score = score;
                max_pointer = p;
            }
        }

        output.clear();
        while (true) {
            output.push_back(SPAN(lattice[max_pointer]));
            if (lattice[max_pointer].begin == 0) break;

            max_pointer = pointers_[max_pointer];
        }
        reverse(output.begin(), output.end());
    }

};


int main() {
    vector<string> raws;
    vector<vector<size_t>> offs;
    vector<vector<labelled_span_t>> spans;
    load<labelled_span_t>(string("dev.tag"), raws, offs, spans);

    vector<string> test_raws;
    vector<vector<size_t>> test_offs;
    vector<vector<labelled_span_t>> test_spans;
    load<labelled_span_t>(string("test.tag"), test_raws, test_offs, test_spans);


    //Indexer<string> tag_indexer;
    auto tag_indexer = make_shared<Indexer<string>>();
    for (size_t i = 0; i < spans.size(); i++) {
        for (size_t j = 0; j < spans[i].size(); j++) {
            tag_indexer->get(spans[i][j].label());
        }
    }
    printf("%lu\n", tag_indexer->size());


    Weight model;
    Weight ave;
    LabelledFeature<labelled_span_t> feature;
    feature.set_tag_indexer(tag_indexer);
    Learner learner;
    Lattice_Generator lg;
    lg.set_tag_indexer(tag_indexer);
    PathFinder pf;

    auto dictionary = make_shared<Dictionary>();
    dictionary->load("tyc.dict");
    feature.set_dictionary(dictionary);

    Eval<labelled_span_t> eval;

    vector<labelled_span_t> lattice;
    vector<labelled_span_t> output;
    Weight gradient;
    size_t iterations = 5;
    for (size_t it = 0; it < iterations; it ++) {
        feature.set_dict(model);
        eval.reset();
        for (size_t i = 0; i < raws.size(); i++) {
            //printf("%s\n", raws[i].c_str());
            //fprintf(stderr, "[%lu/%lu]\n", i, raws.size());
            if (i % 100 == 0) {
                fprintf(stderr, "[%lu/%lu]\r", i, raws.size());
            }
            lg.gen(raws[i], offs[i], spans[i], lattice);
            pf.find_path(raws[i], offs[i], feature, lattice, output);
            gradient.clear();

            feature.calc_gradient(spans[i], output, gradient);
            learner.update(model, gradient);

            eval.eval(spans[i], output);
        }
        eval.report();

        learner.average(model, ave);
        eval.reset();
        feature.set_dict(ave);
        for (size_t i = 0; i < test_raws.size(); i++) {
            lg.gen(test_raws[i], test_offs[i], test_spans[i], lattice);
            pf.find_path(test_raws[i], test_offs[i], feature, lattice, output);
            eval.eval(test_spans[i], output);
        }
        eval.report();
    }
    return 0;
};
