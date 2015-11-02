#pragma once
#include<string>
#include<set>
#include<vector>
#include<memory>

namespace tenseg {
using namespace std;
using namespace google;

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



template<typename SPAN>
struct lattice_t {
    vector<SPAN> spans;
    shared_ptr<string> raw;
    shared_ptr<vector<size_t>> off;
};


template<typename SPAN>
ostream& operator<<(ostream & os, const lattice_t<SPAN>& lat) {
    for (size_t i = 0; i < lat.spans.size(); i++) {
        if (i > 0) os<<" ";
        auto& span = lat.spans[i];
        size_t begin = (*lat.off)[span.begin];
        size_t end = (*lat.off)[span.end];
        os<<lat.raw->substr(begin, end - begin);
        if (span.label().size()) {
            os<<"_"<<span.label();
        }
    }
    return os;
}

class PathFinder {
public:
    PathFinder() {}

    template <class SPAN, class FEATURE>
    void find_path(lattice_t<SPAN>& lat,
            FEATURE& feature,
            lattice_t<SPAN>& out
            ) {
        const string& raw = *lat.raw;
        const vector<size_t>& off = *lat.off;
        vector<SPAN>& lattice = lat.spans;
        out.spans.clear();
        vector<SPAN>& output = out.spans;
        //cout<<">>>>>"<<lattice.size()<<"<<\n";

        feature.prepare(lat.raw, lat.off, lattice);

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
