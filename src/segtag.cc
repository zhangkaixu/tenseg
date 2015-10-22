#include <cstdio>
#include <algorithm>
#include <string>
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


using namespace tenseg;


/// 定义参数
DEFINE_string(train, "", "Training file");
DEFINE_string(test, "", "Development file");
DEFINE_string(txt_model, "", "Development file");
DEFINE_string(dict, "", "Dict file");
DEFINE_int32(iteration, 5, "Iteration");

int main(int argc, char* argv[]) {
    /// 参数解析
    google::ParseCommandLineFlags(&argc, &argv, true);
    //google::InitGoogleLogging(argv[0]);

    /// 训练集
    vector<string> raws;
    vector<vector<size_t>> offs;
    vector<vector<labelled_span_t>> spans;
    if (FLAGS_train.size())
        load<labelled_span_t>(FLAGS_train, raws, offs, spans);

    /// 测试集/验证集
    vector<string> test_raws;
    vector<vector<size_t>> test_offs;
    vector<vector<labelled_span_t>> test_spans;
    if (FLAGS_test.size())
        load<labelled_span_t>(FLAGS_test, test_raws, test_offs, test_spans);


    /// 标签集
    auto tag_indexer = make_shared<Indexer<string>>();
    if (FLAGS_train.size()) {
        for (size_t i = 0; i < spans.size(); i++)
            for (size_t j = 0; j < spans[i].size(); j++)
                tag_indexer->get(spans[i][j].label());
    } else if (FLAGS_txt_model.size()) {
        tag_indexer->load(FLAGS_txt_model + ".tags");
    } else {
        fprintf(stderr, "you should at least specify train or model\n");
        return 0;
    }
    //LOG(INFO)<<"tagset size: "<<tag_indexer->size()<<"\n";
    fprintf(stderr, "tagset size: %lu\n", tag_indexer->size());


    /// 特征
    LabelledFeature<labelled_span_t> feature;
    feature.set_tag_indexer(tag_indexer);
    /// 

    LatticeGenerator lg;
    lg.set_tag_indexer(tag_indexer);

    PathFinder pf;
    Learner<Weight> learner;
    Weight ave;
    if ((!FLAGS_train.size())
            && (FLAGS_txt_model.size())) {
        learner.weight().load(FLAGS_txt_model + ".weights");
    }

    /// 外部词典
    if (FLAGS_dict.size()) {
        auto dictionary = make_shared<Dictionary>();
        dictionary->load(FLAGS_dict.c_str());
        feature.set_dictionary(dictionary);
    }

    Eval<labelled_span_t> eval;

    vector<labelled_span_t> lattice;
    vector<labelled_span_t> output;

    /// 训练模式
    if (FLAGS_train.size()) {
        size_t iterations = FLAGS_iteration;
        for (size_t it = 0; it < iterations; it ++) {
            feature.set_weight(learner.weight());
            eval.reset();
            for (size_t i = 0; i < raws.size(); i++) {
                if (i % 100 == 0) {
                    fprintf(stderr, "[%lu/%lu]\r", i, raws.size());
                }
                lg.gen(raws[i], offs[i], spans[i], lattice);
                pf.find_path(raws[i], offs[i], feature, lattice, output);

                /// update
                Weight gradient;
                feature.calc_gradient(spans[i], output, gradient);
                learner.update(gradient);

                eval.eval(spans[i], output);
            }
            eval.report();

            if (!FLAGS_test.size()) continue;

            learner.average(ave);
            feature.set_weight(ave);
            eval.reset();
            for (size_t i = 0; i < test_raws.size(); i++) {
                lg.gen(test_raws[i], test_offs[i], test_spans[i], lattice);
                pf.find_path(test_raws[i], test_offs[i], feature, lattice, output);
                eval.eval(test_spans[i], output);
            }
            eval.report();
        }

        /// save model if a model name is given
        if (FLAGS_txt_model.size()) {
            learner.average(ave);
            fprintf(stderr, "saving weights and tags\n");
            ave.dump((FLAGS_txt_model + ".weights").c_str());
            tag_indexer->dump((FLAGS_txt_model + ".tags").c_str());
        }
        return 0;
    }
    if (FLAGS_test.size()) { /// 测试模式
        //learner.weight().load(FLAGS_txt_model + ".weights");
        feature.set_weight(learner.weight());
        eval.reset();
        for (size_t i = 0; i < test_raws.size(); i++) {
            lg.gen(test_raws[i], test_offs[i], test_spans[i], lattice);
            pf.find_path(test_raws[i], test_offs[i], feature, lattice, output);
            eval.eval(test_spans[i], output);
        }
        eval.report();
        return 0;
    }

    return 0;
};
