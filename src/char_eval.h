#pragma once
#include <ctime>
namespace tenseg{

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
    void eval(vector<size_t>& tags, vector<size_t>& gold,
            bool dbg = false) {
        bool begin_fit = true;
        if (dbg) {
            for (size_t i = 0; i < tags.size(); i++) {
                printf("%lu ", tags[i]);
            }printf("\n");
            for (size_t i = 0; i < tags.size(); i++) {
                printf("%lu ", gold[i]);
            }printf("\n");
        }
        for (size_t i = 0; i < tags.size(); i++) {
            bool rst_flag = false;
            if (((i + 1) == tags.size())
                    || (tags[i] > 1)) {
                _rst++;
                rst_flag = true;
            }

            bool std_flag = false;
            if (((i + 1) == gold.size())
                    || (gold[i] > 1)) {
                _std++;
                std_flag = true;
            }
            if (std_flag != rst_flag) {
                begin_fit = false;
            }
            if (std_flag and rst_flag and begin_fit) {
                _cor++;
            }
            if (std_flag and rst_flag) {
                begin_fit = true;
            }
        }
        if (dbg) report();
    }
    void report() {
        double p = 1.0 * _cor / _rst;
        double r = 1.0 * _cor / _std;
        double f = 2 * p * r / (p + r);
        //std::time(&_end_time);
        _end_time = std::clock();
        std::cout << _end_time << " ";
        printf("%lu %lu %lu %.3g %.3g %.3g %.3g\n", _std, _rst, _cor,
                p, r, f, ((double)(_end_time - _start_time) / CLOCKS_PER_SEC)
                );
    }
};
}
