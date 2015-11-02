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


#ifdef Debug
#define LOG_INFO(x) LOG(INFO) << x
#else
#define LOG_INFO(x)
#endif

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

size_t unicode(const char* p, const size_t len) {
   size_t code = 0;
    switch (len) {
        case 0:
           return 0;
        case 1:
           return (*p) & 0x7F;
        case 2:
           code = *(p++) & 0x1F;
           code = (code << 6) | (*(p++) & 0x3F);
           return code;
        case 3:
           code = *(p++) & 0x0F;
           code = (code << 6) | (*(p++) & 0x3F);
           code = (code << 6) | (*(p++) & 0x3F);
           return code;
    }
    return 0;
}

void to_half(const string& src_raw,
        const vector<size_t>& src_off,
        string& tgt_raw,
        vector<size_t>& tgt_off) {

    vector<char> buffer;
    buffer.reserve(src_raw.size());
    tgt_off.clear();
    tgt_off.push_back(0);

    for (size_t i = 0; i < src_off.size() - 1; i++) {
        size_t begin = src_off[i];
        size_t end = src_off[i+1];
        //printf("%s\n", src_raw.substr(src_off[i], src_off[i+1] - src_off[i]).c_str());
        size_t code = unicode(src_raw.data() + src_off[i], src_off[i+1] - src_off[i]);
        //printf("%lu\n", unicode(src_raw.data() + src_off[i], src_off[i+1] - src_off[i]));

        if (code >= 0xff01 && code <= 0xff5e) {
            //printf("%s\n", src_raw.substr(src_off[i], src_off[i+1] - src_off[i]).c_str());
            buffer.push_back(code - 65248);
            //printf("%c\n", code - 65248);
        } else {
            for (size_t j = begin; j < end; j++) {
                buffer.push_back(src_raw[j]);
            }
        }
        tgt_off.push_back(buffer.size());
    }

    buffer.push_back(0);
    tgt_raw = string(buffer.data());
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

}
