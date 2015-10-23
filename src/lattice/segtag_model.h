#pragma once
#include "lattice/lattice.h"
#include "lattice/feature.h"

namespace tenseg {
using namespace std;
template<typename SPAN>
class SegTag {
public:
    SegTag() {
        tag_indexer_ = make_shared<Indexer<string>>();
        feature_.set_tag_indexer(tag_indexer_);
    }

    template<class LG>
    void fit(
            vector<lattice_t<SPAN>>& train_Xs,
            vector<lattice_t<SPAN>>& train_Ys,
            vector<lattice_t<SPAN>>& test_Xs,
            vector<lattice_t<SPAN>>& test_Ys,
            LG& lg,
            size_t iterations
            ) {

        for (auto& lattice : train_Xs) {
            for (auto& span : lattice.spans) {
                tag_indexer_->get(span.label());
            }
        }

        Eval<SPAN> eval;
        Learner<Weight> learner;
        lattice_t<SPAN> out;

        for (size_t it = 0; it < iterations; it ++) {
            feature_.set_weight(learner.weight());
            eval.reset();
            for (size_t i = 0; i < train_Xs.size(); i++) {
                if (i % 100 == 0) {
                    fprintf(stderr, "[%lu/%lu]\r", i, train_Xs.size());
                }
                lg.gen(train_Xs[i]);
                decoder_.find_path(train_Xs[i], feature_, out);
                train_Xs[i].spans.clear();
                train_Xs[i].spans.shrink_to_fit();
                /// update
                Weight gradient;
                feature_.calc_gradient(train_Ys[i].spans, out.spans, gradient);
                learner.update(gradient);

                eval.eval(train_Ys[i].spans, out.spans);
            }
            eval.report();

            if (!test_Xs.size()) continue;

            learner.average(ave);
            feature_.set_weight(ave);
            eval.reset();
            for (size_t i = 0; i < test_Xs.size(); i++) {
                lg.gen(test_Xs[i]);
                decoder_.find_path(test_Xs[i], feature_, out);
                eval.eval(test_Ys[i].spans, out.spans);
                test_Xs[i].spans.clear();
                test_Xs[i].spans.shrink_to_fit();
            }
            eval.report();
        }

        learner.average(ave);
        feature_.set_weight(ave);
    }

    template<class LG>
    void predict(vector<lattice_t<SPAN>>& test_Xs,
            vector<lattice_t<SPAN>>& test_Ys,
            LG& lg
            ) {
        test_Ys.clear();
        for (size_t i = 0; i < test_Xs.size(); i++) {
            test_Ys.emplace(test_Ys.end());
            lg.gen(test_Xs[i]);
            decoder_.find_path(test_Xs[i], feature_, test_Ys.back());
            test_Ys.back().raw = test_Xs[i].raw;
            test_Ys.back().off = test_Xs[i].off;
        }
    }
    template<class LG>
    void test(vector<lattice_t<SPAN>>& test_Xs,
            vector<lattice_t<SPAN>>& test_Ys,
            LG& lg
            ) {
        Eval<SPAN> eval;
        eval.reset();
        lattice_t<SPAN> out;
        for (size_t i = 0; i < test_Xs.size(); i++) {
            lg.gen(test_Xs[i]);
            decoder_.find_path(test_Xs[i], feature_, out);
            eval.eval(test_Ys[i].spans, out.spans);
            test_Xs[i].spans.clear();
            test_Xs[i].spans.shrink_to_fit();
        }
        eval.report();
    }
    LabelledFeature<SPAN>& feature() {
        return feature_;
    }
    shared_ptr<Indexer<string>>& tag_indexer() {
        return tag_indexer_;
    }
    void save(const string& txt_model) {
        fprintf(stderr, "saving weights and tags\n");
        ave.dump((txt_model + ".weights").c_str());
        tag_indexer_->dump((txt_model + ".tags").c_str());
    }
    void load(const string& txt_model) {
        ave.load(txt_model + ".weights");
        tag_indexer_->load(txt_model + ".tags");
        feature_.set_weight(ave);
    }

private:
    shared_ptr<Indexer<string>> tag_indexer_;
    PathFinder decoder_;
    LabelledFeature<SPAN> feature_;
    Weight ave;
};
}
