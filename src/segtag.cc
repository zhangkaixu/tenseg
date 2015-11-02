#include "gflags/gflags.h"
#include "glog/logging.h"

#include "common/common.h"
#include "common/weight.h"
#include "common/optimizer.h"
#include "common/dictionary.h"

#include "lattice/segtag_model.h"

#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>


using namespace tenseg;


class LatticeGenerator {
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
    LatticeGenerator() {
        _punc.insert(string("。")); _punc.insert(string("，"));
        _punc.insert(string("？")); _punc.insert(string("！"));
        _punc.insert(string("：")); _punc.insert(string("“"));
        _punc.insert(string("”"));
    }

    void set_tag_indexer(shared_ptr<Indexer<string>> ti) {
        _tag_indexer = ti;
    }

    void gen(lattice_t<labelled_span_t>& lat) {
        const string& raw = *lat.raw;
        const vector<size_t>& off = *lat.off;
        vector<labelled_span_t>& lattice = lat.spans;

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


template<class C>
void utf8_off(const C& raw, vector<size_t>& off) {
    off.clear();
    for (size_t i = 0; i < raw.size(); i++) {
        const char& c = raw[i];
        if ((0xc0 == (c & 0xc0))
                || !(c & 0x80)) {
            off.push_back(i);
        }
    }
    off.push_back(raw.size());
}


/**
 * load corpus from a segmented file
 * */
template<class SPAN>
void load(
        const string& filename,
        shared_ptr<Indexer<string>> tag_indexer,
        vector<lattice_t<SPAN>>& Xs,
        vector<lattice_t<SPAN>>& Ys
        ){

    std::ifstream input(filename);
    
    for (std::string line; std::getline(input, line); ) {
        Xs.push_back(lattice_t<SPAN>());
        Ys.push_back(lattice_t<SPAN>());
        vector<SPAN>& sent = Ys.back().spans;

        size_t offset = 0;
        std::istringstream iss(line);
        std::string item;
        vector<char> raw;
        while (!iss.eof()) {
            iss >> item;
            sent.push_back(SPAN(item, offset, raw));
        }

        Ys.back().off = make_shared<vector<size_t>>();
        vector<size_t>& off = *Ys.back().off;
        utf8_off(raw, off);

        raw.push_back(0);
        Ys.back().raw = make_shared<string>(&raw[0]);
        Xs.back().raw = Ys.back().raw;
        Xs.back().off = Ys.back().off;
    }

    if (tag_indexer->size() == 0) {
        for (auto& x : Ys) {
            for (auto& span : x.spans) {
                tag_indexer->get(span.label());
            }
        }
    }
};


/// 定义参数
DEFINE_string(train, "", "Training file");
DEFINE_string(test, "", "Development file");
DEFINE_string(txt_model, "", "Development file");
DEFINE_string(dict, "", "Dict file");
DEFINE_string(phrase, "", "phrase Dict file");
DEFINE_int32(iteration, 5, "Iteration");
//DEFINE_int32(logtostderr, 1, "");

int main(int argc, char* argv[]) {
    typedef labelled_span_t span_type;
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 2;
    FLAGS_minloglevel = 1;
    
    /// 命令行参数解析
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    /// 模型
    SegTag<span_type> segtag;

    /// 语料
    vector<lattice_t<span_type>> train_Xs;
    vector<lattice_t<span_type>> train_Ys;
    vector<lattice_t<span_type>> test_Xs;
    vector<lattice_t<span_type>> test_Ys;

    /// 外部词典
    if (FLAGS_dict.size()) {
        for (auto& dfile : split(FLAGS_dict, ',')) {
            auto df = make_shared<DictFeature<span_type>>(dfile);
            segtag.feature().features().push_back(df);
        }
    }

    if (FLAGS_phrase.size()) {
        for (auto& dfile : split(FLAGS_phrase, ',')) {
            auto df = make_shared<PhraseFeature<span_type>>(dfile);
            segtag.feature().features().push_back(df);
        }
    }

    /// 词图产生
    LatticeGenerator lg;
    lg.set_tag_indexer(segtag.tag_indexer());

    /// load
    if ((!FLAGS_train.size()) && (FLAGS_txt_model.size())) {
        segtag.load(FLAGS_txt_model);
    }

    /// 训练模式
    if (FLAGS_train.size()) {
        load(FLAGS_train, segtag.tag_indexer(), train_Xs, train_Ys);
        if (FLAGS_test.size()) {
            load(FLAGS_test, segtag.tag_indexer(), test_Xs, test_Ys);
        }

        size_t iterations = FLAGS_iteration;
        segtag.fit(train_Xs, train_Ys, test_Xs, test_Ys, lg, iterations);

        if (FLAGS_txt_model.size()) {
            segtag.save(FLAGS_txt_model);
        }
        return 0;
    }

    /// 测试模式
    if (FLAGS_test.size()) { 
        load(FLAGS_test, segtag.tag_indexer(), test_Xs, test_Ys);
        segtag.test(test_Xs, test_Ys, lg);
        return 0;
    }

    /// 预测模式
    if (FLAGS_txt_model.size()) {
        vector<lattice_t<span_type>> Xs(1);
        vector<lattice_t<span_type>> Ys(1);
        Xs.back().raw = make_shared<string>();
        Xs.back().off = make_shared<vector<size_t>>();

        
        for (; std::getline(cin, *Xs.back().raw); ) {
            utf8_off(*Xs.back().raw, *Xs.back().off);
            segtag.predict(Xs, Ys, lg);
            cout << Ys.back() << endl;
        }
    }

    return 0;
};
