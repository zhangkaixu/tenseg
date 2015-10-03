#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "tenseg.h"
#include "feature.h"
#include "weight.h"
#include "optimizer.h"
#include "dictionary.h"

using namespace std;

using namespace tenseg;


class Lattice_Generator {
public:
    void gen(const string& raw, 
            const vector<size_t>& off, 
            const vector<span_t>& span,
            vector<linked_span_t>& lattice) {

        if (off.size() == 0) return;
        size_t n = off.size() - 1;

        lattice.clear();
        // generate all spans
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n + 1; j++) {
                if (j - i > 10) break;
                lattice.push_back(linked_span_t(i, j));
            }
        }
    }
};

void gen_lattice(const string& raw, 
        const vector<size_t>& off, 
        const vector<span_t>& span,
        vector<linked_span_t>& lattice) {

    if (off.size() == 0) return;
    size_t n = off.size() - 1;

    lattice.clear();
    // generate all spans
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n + 1; j++) {
            if (j - i > 10) break;
            lattice.push_back(linked_span_t(i, j));
        }
    }
}



void find_path(const string& raw,
        const vector<size_t>& off,
        Feature& feature,
        vector<linked_span_t>& lattice,
        vector<span_t>& output
        ) {
    
    feature.prepare(raw, off, lattice);
    /// Step 1 build path
    vector<vector<size_t>> begins;
    vector<vector<size_t>> ends;
    for (size_t i = 0; i < off.size(); i++) {
        begins.push_back(vector<size_t>());
        ends.push_back(vector<size_t>());
    }

    for (size_t i = 0; i < lattice.size(); i++) {
        ends[lattice[i].end].push_back(i);
        begins[lattice[i].begin].push_back(i);
    }

    /// Step 2 search
    for (size_t i = 0; i < off.size() - 1; i++) {
        //printf("[%lu]= %lu\n", i, begins[i].size());
        for (size_t k = 0; k < begins[i].size(); k++) {
            linked_span_t& this_span = lattice[begins[i][k]];

            double max_score = 0;
            size_t max_pointer = 0;
            bool has_max = false;
            for (size_t j = 0; j < ends[i].size(); j ++) {
                double score = 0;
                size_t p = ends[i][j];
                score = lattice[p].score;
                score += feature.bigram(ends[i][j], begins[i][k]);
                /// bigram features
                if (!has_max || max_score < score) {
                    has_max = true;
                    max_score = score;
                    max_pointer = p;
                }
            }
            /// unigram features
            this_span.score = max_score;
            this_span.score += feature.unigram(begins[i][k]);
            this_span.pointer = max_pointer;
        }
    }

    /// Step 3 find best
    double max_score = 0;
    size_t max_pointer = 0;
    bool has_max = false;
    for (size_t j = 0; j < ends.back().size(); j ++) {
        double score = 0;
        size_t p = ends.back()[j];
        score = lattice[p].score;
        if (!has_max || max_score < score) {
            has_max = true;
            max_score = score;
            max_pointer = p;
        }
    }

    output.clear();
    while (true) {
        output.push_back(span_t(lattice[max_pointer].begin, 
                lattice[max_pointer].end));
        if (lattice[max_pointer].begin == 0) break;
        max_pointer = lattice[max_pointer].pointer;
    }
    reverse(output.begin(), output.end());
}



int main() {
    vector<string> raws;
    vector<vector<size_t>> offs;
    vector<vector<span_t>> spans;
    load_corpus(string("train.seg"), raws, offs, spans);

    vector<string> test_raws;
    vector<vector<size_t>> test_offs;
    vector<vector<span_t>> test_spans;
    load_corpus(string("test.seg"), test_raws, test_offs, test_spans);

    Weight model;
    Weight ave;
    Feature feature;
    Learner learner;

    auto dictionary = make_shared<Dictionary>();
    dictionary->load("tyc.dict");

    feature.set_dictionary(dictionary);

    Eval eval;

    vector<linked_span_t> lattice;
    vector<span_t> output;
    Weight gradient;
    for (size_t it = 0; it < 5; it ++) {
        feature.set_dict(model);
        for (size_t i = 0; i < raws.size(); i++) {
            gen_lattice(raws[i], offs[i], spans[i], lattice);
            find_path(raws[i], offs[i], feature, lattice, output);
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
            gen_lattice(test_raws[i], test_offs[i], test_spans[i], lattice);
            find_path(test_raws[i], test_offs[i], feature, lattice, output);
            eval.eval(test_spans[i], output);
        }
        eval.report();
    }
    return 0;
};
