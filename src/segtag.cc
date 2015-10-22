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
DEFINE_string(train, "train.tag", "Training file");
DEFINE_string(dev, "test.tag", "Development file");
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
    load<labelled_span_t>(FLAGS_train, raws, offs, spans);

    /// 测试集/验证集
    vector<string> test_raws;
    vector<vector<size_t>> test_offs;
    vector<vector<labelled_span_t>> test_spans;
    load<labelled_span_t>(FLAGS_dev, test_raws, test_offs, test_spans);


    /// 标签集
    auto tag_indexer = make_shared<Indexer<string>>();
    for (size_t i = 0; i < spans.size(); i++) {
        for (size_t j = 0; j < spans[i].size(); j++) {
            tag_indexer->get(spans[i][j].label());
        }
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
    Learner learner;
    Weight model;
    Weight ave;

    /// 外部词典
    if (FLAGS_dict.size()) {
        auto dictionary = make_shared<Dictionary>();
        dictionary->load(FLAGS_dict.c_str());
        feature.set_dictionary(dictionary);
    }

    Eval<labelled_span_t> eval;

    vector<labelled_span_t> lattice;
    vector<labelled_span_t> output;
    size_t iterations = FLAGS_iteration;
    for (size_t it = 0; it < iterations; it ++) {
        feature.set_weight(model);
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
