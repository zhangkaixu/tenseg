#pragma once
#include <map>
#include <vector>
#include <string>
#include <sstream>
/**
 * a dict of {string : [double]}
 * */
namespace dict {
using std::map;
using std::string;
using std::vector;

class Learner;

class Dict {
private:
    vector<double> _data;
    vector<size_t> _ends;
    map<string, size_t> _map;
public:
    void clear() {
        _data.clear();
        _ends.clear();
        _map.clear();
    }
    Dict() {
    }
    /**create a dict associate with a file*/
    Dict(const char* filename) {
    }
    void load(const char* filename) {
        std::ifstream input(filename);
        string str;
        vector<double> vec;
        double v;
        for (std::string line; std::getline(input, line); ) {
            std::istringstream iss(line);
            vec.clear();
            iss >> str;
            while (!iss.eof()) {
                iss >> v;
                vec.push_back(v);
            }
            //printf("%s %lu\n", str.c_str(), vec.size());
            this->add_from(str, &(vec[0]), vec.size());
        }
    }
    void dump(const char* filename) {
        std::FILE* pf = fopen(filename, "w");
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            fprintf(pf, "%s", it->first.c_str());
            size_t end_ind = it->second;
            size_t begin = ((end_ind > 0)?_ends[end_ind - 1]:0);
            size_t end = _ends[end_ind];
            for (size_t i = begin; i < end; i++) {
                fprintf(pf, "\t%g", _data[i]);
            }fprintf(pf, "\n");
        }
    }

    void dbg(const string& key) {
        double* ptr;
        size_t len;
        get(key, ptr, len);
        printf("[%s]", key.c_str());
        for (size_t i = 0; i < len; i++) {
            printf(" %g", ptr[i]);
        }
        printf("\n");
    }

    void insert(const string& key, const size_t length){
        // check existing
        auto result = _map.find(key);
        if (result != _map.end()) {
            return;
        }
        // memory
        _map[key] = _ends.size();
        for (size_t i = 0; i < length; i++) {
            _data.push_back(0);
        }
        _ends.push_back(_data.size());
    };
    void get(const string& key, double*& ptr, size_t& len) {
        ptr = nullptr;
        len = 0;
        auto result = _map.find(key);
        if (result == _map.end()) {
            return;
        }
        size_t end_off = result->second;
        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
        size_t end = _ends[end_off];
        ptr = &(_data[0]) + begin;
        len = end - begin;
    }
    double* get(const string& key) {
        auto result = _map.find(key);
        if (result == _map.end()) {
            return nullptr;
        }
        size_t end_off = result->second;
        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
        return &(_data[0]) + begin;
    }
    void add_from(const string& key, const double* ptr, 
            const size_t len, double eta = 1.0) {
        bool is_all_zero = true;
        for (size_t i = 0; i < len; i++) {
            if (ptr[i]) {
                is_all_zero = false;
                break;
            }
        }
        if (is_all_zero) return;

        auto result = _map.find(key);
        if (result == _map.end()) {
            insert(key, len);
        }

        double* m = get(key);
        for (size_t i = 0; i < len; i++) {
            m[i] += ptr[i] * eta;
        }
    }
    void add_to(const string& key, double* ptr) {
        auto result = _map.find(key);
        if (result == _map.end()) {
            return;
        }
        size_t end_off = result->second;
        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
        size_t end = _ends[end_off];
        for (size_t i = begin; i < end; i++) {
            ptr[i - begin] += _data[i];
        }
    }
    void update(Dict& other, double eta) {
        for (auto it = other._map.begin();
                it != other._map.end();
                ++ it) {
            double* ptr;
            size_t len;
            other.get(it->first, ptr, len);
            this->add_from(it->first, ptr, len, eta);
        }
    }
};

class Learner {
private:
    Dict _acc;
    size_t _step;
public:
    Learner() {
        _step = 0;
    }
    void update(Dict& model, Dict& gradient) {
        _step++;
        model.update(gradient, 1.0);
        _acc.update(gradient, _step);
    }
    void average(Dict& model, Dict& ave) {
        ave.clear();
        ave.update(model, 1.0);
        ave.update(_acc, - 1.0 / _step);
    }
};

}
