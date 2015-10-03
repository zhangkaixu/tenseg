#pragma once
#include <vector>

namespace tenseg {

using std::vector;

void viterbi(const size_t N, vector<double>& transition, vector<double>& emission, 
        vector<size_t>& tags) {
    vector<double> score;
    vector<size_t> pointer;

    // init the first node
    for (size_t i = 0; i < N; i++) {
        score.push_back(emission[i]);
        pointer.push_back(0);
    }

    // forward
    size_t i = 1;
    for (; i < emission.size() / N; i++) { // for each node
        for (size_t j = 0; j < N; j++) {
            size_t k = 0;
            double best_score = score[(i - 1) * N + k] + transition[k * N + j];
            size_t best_index = k;
            for (k = 1; k < N; k++) {
                double this_score = score[(i - 1) * N + k] 
                    + transition[k * N + j];
                if (this_score > best_score) {
                    best_score = this_score;
                    best_index = k;
                }
            }
            score.push_back(best_score + emission[i * N + j]);
            pointer.push_back(best_index);
        }
    }
    // backtrack
    size_t k = 0;
    double best_score = score[(i - 1) * N + k];
    size_t best_index = k;
    for (k = 1; k < N; k++) {
        double this_score = score[(i - 1) * N + k];
        if (this_score > best_score) {
            best_score = this_score;
            best_index = k;
        }
    }
    tags.clear();
    for (size_t j = 0; j < emission.size() / N; j++) {
        tags.push_back(0);
    }

    while (true) {
        if (i == 0) break;
        i--;
        tags[i] = best_index;
        best_index = pointer[i * N + best_index];
    }
}

void word_dp(vector<double>& transition, vector<double>& emission,
        vector<size_t>& tags) {
}

};
