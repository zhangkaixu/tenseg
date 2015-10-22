#pragma once
#include<string>
#include<vector>

namespace tenseg {
using namespace std;

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
} // namespace
