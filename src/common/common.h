#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <ctime>

namespace tenseg {
using namespace std;

/**
 * decl
 * */


/// implementations

template <class SPAN>
class Eval {
private:
    size_t _std;
    size_t _rst;
    size_t _cor;
    size_t _label_cor;
    time_t _start_time;
    time_t _end_time;
public:
    void reset() {
        _std = 0;
        _rst = 0;
        _cor = 0;
        _label_cor = 0;
        _start_time = std::clock();
    }
    Eval() {
        reset();
    }
    void eval(vector<SPAN>& gold, vector<SPAN>& output) {
        _std += gold.size();
        _rst += output.size();

        size_t ind_g = 0;
        size_t ind_o = 0;
        bool begin_fit = true;
        while (true) {
            if (ind_g >= gold.size() || ind_o >= output.size()) break;
            if (gold[ind_g].begin != output[ind_o].begin) {
                if (gold[ind_g].begin > output[ind_o].begin) {
                    ind_o ++;
                } else {
                    ind_g ++;
                }
                continue;
            }
            
            if (gold[ind_g].end != output[ind_o].end) {
                if (gold[ind_g].end > output[ind_o].end) {
                    ind_o++;
                } else {
                    ind_g++;
                }
                continue;
            } else {
                _cor += 1;
                if (gold[ind_g] == output[ind_o]) {
                    _label_cor += 1;
                }
                ind_o++;
                ind_g++;
            }
        }

    }


    void report() {
        double p = 1.0 * _cor / _rst;
        double r = 1.0 * _cor / _std;
        double f = 2 * p * r / (p + r);
        double lf = _get_f(_std, _rst, _label_cor);
        _end_time = std::clock();
        //std::cout << _end_time << " ";
        printf("%lu %lu %lu \033[40;32m%.5g %.5g\033[0m %.3g(sec.)\n", _std, _rst, _cor,
                lf, f, ((double)(_end_time - _start_time) / CLOCKS_PER_SEC)
                );
    }
private:
    double _get_f(double std, double rst, double cor) {
        double p = 1.0 * cor / rst;
        double r = 1.0 * cor / std;
        double f = 2 * p * r / (p + r);
        return f;
    }
};


/**
 * load corpus from a segmented file
 * */
template<class SPAN>
void load(
        const string& filename,
        vector<string>& raws,
        vector<vector<size_t>>& offs,
        vector<vector<SPAN>>& corpus
        ){

    std::ifstream input(filename);
    
    for (std::string line; std::getline(input, line); ) {
        corpus.push_back(vector<SPAN>());
        vector<SPAN>& sent = corpus.back();

        size_t offset = 0;
        std::istringstream iss(line);
        std::string item;
        vector<char> raw;
        while (!iss.eof()) {
            iss >> item;
            //cout<<"["<<item<<"]\n";
            sent.push_back(SPAN(item, offset, raw));
        }

        offs.push_back(vector<size_t>());
        vector<size_t>& off = offs.back();
        for (size_t i = 0; i < raw.size(); i++) {
            const char& c = raw[i];
            if ((0xc0 == (c & 0xc0))
                    || !(c & 0x80)) {
                off.push_back(i);
            }
        }
        off.push_back(raw.size());

        raw.push_back(0);
        raws.push_back(string(&raw[0]));

        //printf("%lu %s\n", raws.back().size(), raws.back().c_str());
        //for (size_t i = 0; i < off.size(); i++) {
        //    printf(" %lu", off[i]);
        //}printf("\n");

    }
};

template <class T>
class Indexer {
public:
    Indexer() {}
    size_t get(const T& ref) {
        auto ret = index_.find(ref);
        if (ret == index_.end()) {
            //printf("new\n");
            size_t ind = index_.size();
            index_[ref] = ind;
            list_.push_back(ref);
            return ind;
        }
        return ret->second;
    }
    string& operator[](size_t ind) {
        return list_[ind];
    }
    size_t size() const {
        return index_.size();
    }

    void dump(const string& filename) const{
        std::ofstream output(filename);
        for (auto iter : list_)
            output << iter << "\n";
        output.close();
    }

    void load(const string& filename) {
        std::ifstream input(filename);
        index_.clear();
        list_.clear();
        for (std::string line; std::getline(input, line); ) {
            index_[line] = list_.size();
            list_.push_back(line);
        }
        //fprintf(stderr, "load %lu tags\n", list_.size());
        input.close();
    }
private:
    map<T, size_t> index_;
    vector<T> list_;
};

}
