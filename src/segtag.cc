#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "gflags/gflags.h"
//#include "glog/logging.h"

#include "common/common.h"
#include "common/weight.h"
#include "common/optimizer.h"
#include "common/dictionary.h"

#include "lattice/lattice.h"
#include "lattice/feature.h"



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




DEFINE_string(train, "train.tag", "Training file");
DEFINE_string(dev, "test.tag", "Development file");
DEFINE_string(txt_model, "", "Development file");
DEFINE_string(dict, "", "Dict file");
DEFINE_int32(iteration, 5, "Iteration");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    //google::InitGoogleLogging(argv[0]);

    vector<string> raws;
    vector<vector<size_t>> offs;
    vector<vector<labelled_span_t>> spans;
    load<labelled_span_t>(FLAGS_train, raws, offs, spans);

    vector<string> test_raws;
    vector<vector<size_t>> test_offs;
    vector<vector<labelled_span_t>> test_spans;
    load<labelled_span_t>(FLAGS_dev, test_raws, test_offs, test_spans);


    //Indexer<string> tag_indexer;
    auto tag_indexer = make_shared<Indexer<string>>();
    for (size_t i = 0; i < spans.size(); i++) {
        for (size_t j = 0; j < spans[i].size(); j++) {
            tag_indexer->get(spans[i][j].label());
        }
    }

    //LOG(INFO)<<"tagset size: "<<tag_indexer->size()<<"\n";
    fprintf(stderr, "tagset size: %lu\n", tag_indexer->size());


    Weight model;
    Weight ave;
    LabelledFeature<labelled_span_t> feature;
    feature.set_tag_indexer(tag_indexer);
    Learner learner;
    Lattice_Generator lg;
    lg.set_tag_indexer(tag_indexer);
    PathFinder pf;

    if (FLAGS_dict.size()) {
        auto dictionary = make_shared<Dictionary>();
        dictionary->load(FLAGS_dict.c_str());
        feature.set_dictionary(dictionary);
    }

    Eval<labelled_span_t> eval;

    vector<labelled_span_t> lattice;
    vector<labelled_span_t> output;
    Weight gradient;
    size_t iterations = FLAGS_iteration;
    for (size_t it = 0; it < iterations; it ++) {
        feature.set_weight(model);
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
        feature.set_weight(ave);
        for (size_t i = 0; i < test_raws.size(); i++) {
            lg.gen(test_raws[i], test_offs[i], test_spans[i], lattice);
            pf.find_path(test_raws[i], test_offs[i], feature, lattice, output);
            eval.eval(test_spans[i], output);
        }
        eval.report();
    }

    /// save model if a model name is given
    if (FLAGS_txt_model.size()) {
        fprintf(stderr, "saving model to %s\n", FLAGS_txt_model.c_str());
        ave.dump(FLAGS_txt_model.c_str());
    }
    return 0;
};