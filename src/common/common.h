#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

namespace tenseg {
using namespace std;

/**
 * decl
 * */

struct span_t {
    size_t begin;
    size_t end;
    span_t(size_t b, size_t e) : begin(b), end(e) {
    }
    template <class T>
    span_t(const T& ref) {
        begin = ref.begin;
        end = ref.end;
    }

    //span_t(const char* s, size_t l, size_t b) : begin(b), end(b+l) {
    //}

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
};

//struct linked_span_t : public span_t {
//    linked_span_t(size_t b, size_t e) :
//        span_t(b, e), score(0), pointer(0)
//    {};
//    double score;
//    size_t pointer;
//};

class Eval;
void load_corpus( const string& filename,
        vector<string>& raws, vector<vector<size_t>>& offs, vector<vector<span_t>>& spans);



/// implementations

class Eval {
private:
    size_t _std;
    size_t _rst;
    size_t _cor;
    time_t _start_time;
    time_t _end_time;
public:
    void reset() {
        _std = 0;
        _rst = 0;
        _cor = 0;
        _start_time = std::clock();
    }
    Eval() {
        reset();
    }
    void eval(vector<span_t>& gold, vector<span_t>& output) {
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
                ind_o++;
                ind_g++;
            }
        }

    }
    void report() {
        double p = 1.0 * _cor / _rst;
        double r = 1.0 * _cor / _std;
        double f = 2 * p * r / (p + r);
        _end_time = std::clock();
        //std::cout << _end_time << " ";
        printf("%lu %lu %lu %.3g %.3g \033[40;32m%.5g\033[0m %.3g(sec.)\n", _std, _rst, _cor,
                p, r, f, ((double)(_end_time - _start_time) / CLOCKS_PER_SEC)
                );
    }
};

span_t parse_item(std::string& str, size_t& offset, vector<unsigned char>& raw) {
    span_t span(offset, offset + str.size());
    offset += str.size();
    return span;
}

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


}
