#pragma once
#include<string>
#include<set>
#include<vector>

namespace tenseg {
using namespace std;

struct span_t {
    size_t begin;
    size_t end;
    span_t() : begin(0), end(0) {};
    span_t(size_t b, size_t e) : begin(b), end(e) {
    }
    template <class T>
    span_t(const T& ref) {
        begin = ref.begin;
        end = ref.end;
    }

    static const string default_label;
    const string& label() const{
        return default_label;
    }

    span_t(std::string& str, size_t& offset, vector<char>& raw) {
        begin = offset;
        for (size_t i = 0; i < str.size(); i++) {
            const char& c = str[i];
            raw.push_back(c);
            if ((0xc0 == (c & 0xc0))
                    || !(c & 0x80)) {
                offset++;
            }
        }
        end = offset;
    }
    bool operator==(span_t& other)const {
        if (begin != other.begin) return false;
        if (end != other.end) return false;
        return true;
    }
};

struct labelled_span_t : public span_t {
    string label_;

    labelled_span_t(size_t b, size_t e) : span_t(b, e), label_(string()){};
    labelled_span_t(size_t b, size_t e, string& l) : span_t(b, e), label_(l){};

    const string& label() const{
        return label_;
    }
    labelled_span_t(std::string& str, size_t& offset, vector<char>& raw) {
        begin = offset;
        for (size_t i = 0; i < str.size(); i++) {
            const char& c = str[i];
            if (c == '_') {
                label_ = str.substr(i + 1);
                break;
            }
            raw.push_back(c);
            if ((0xc0 == (c & 0xc0))
                    || !(c & 0x80)) {
                offset++;
            }
        }
        end = offset;
    }
    bool operator==(labelled_span_t& other)const {
        if (begin != other.begin) return false;
        if (end != other.end) return false;
        if (label_ != other.label_) return false;
        return true;
    }
};

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
public:
    PathFinder() {}

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
        scores_.insert(scores_.end(), lattice.size(), 0);
        pointers_.insert(pointers_.end(), lattice.size(), 0);

        for (size_t i = 0; i < lattice.size(); i++) {
            ends[lattice[i].end].push_back(i);
            begins[lattice[i].begin].push_back(i);
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

private:
    vector<vector<size_t>> begins;
    vector<vector<size_t>> ends;

    vector<double> scores_;
    vector<size_t> pointers_;

};
} // namespace
