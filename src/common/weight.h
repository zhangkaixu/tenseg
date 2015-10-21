#pragma once
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
/**
 * a dict of {string : [double]}
 * */
namespace tenseg {
using std::map;
using std::string;
using std::vector;


class IWeight {
public:
    virtual void load(const char* filename) = 0;
    virtual void dump(const char* filename) = 0;

    virtual void clear() = 0;
    virtual void insert(const string& key, const size_t length) = 0;
    virtual void get(const string& key, double*& ptr, size_t& len) = 0;
    virtual double* get(const string& key) = 0;
    virtual void add_from(const string& key, const double* ptr, 
            const size_t len, double eta = 1.0) = 0;

    virtual void add_to(const string& key, double* ptr) = 0;
    virtual void update(IWeight& other, double eta) = 0;

};

class Weight {
private:
    map<string, vector<double>> _map;
public:
    void clear() {
        _map.clear();
    }
    Weight() {
    }
    /**create a dict associate with a file*/
    Weight(const char* filename) {
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
            this->add_from(str, &(vec[0]), vec.size());
        }
    }
    void dump(const char* filename) {
        //std::FILE* pf = fopen(filename, "w");
        //for (auto it = _map.begin();
        //        it != _map.end();
        //        ++ it) {
        //    fprintf(pf, "%s", it->first.c_str());
        //    size_t end_ind = it->second;
        //    size_t begin = ((end_ind > 0)?_ends[end_ind - 1]:0);
        //    size_t end = _ends[end_ind];
        //    for (size_t i = begin; i < end; i++) {
        //        fprintf(pf, "\t%g", _data[i]);
        //    }fprintf(pf, "\n");
        //}
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
        /// check existing
        auto result = _map.find(key);
        if (result != _map.end()) {
            /// exists, do not insert
            return;
        }
        /// create new item
        vector<double>& vec = _map[key];
        vec.reserve(length);
        for (size_t i = 0; i < length; i++) {
            vec.push_back(0);
        }
        vec.reserve(length);
    };
    void get(const string& key, double*& ptr, size_t& len) {
        ptr = nullptr;
        len = 0;
        auto result = _map.find(key);
        if (result == _map.end()) {
            return;
        }
        vector<double>& vec = _map[key];
        len = vec.size();
        ptr = &vec[0];
    }

    double* get(const string& key) {
        auto result = _map.find(key);
        if (result == _map.end()) {
            return nullptr;
        }
        vector<double>& vec = _map[key];
        return &vec[0];
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

        vector<double>& vec = _map[key];

        for (size_t i = 0; i < vec.size(); i++) {
            ptr[i] += vec[i];
        }
    }
    void update(Weight& other, double eta) {
        for (auto it = other._map.begin();
                it != other._map.end();
                ++ it) {
            double* ptr;
            size_t len;
            other.get(it->first, ptr, len);
            /// if all zero, do nothing
            /// this is especially helpful for memory-saving
            if (std::all_of(ptr, ptr+len, [](double x){return x==0;})) {
                continue;
            }
            this->add_from(it->first, ptr, len, eta);
        }
    }
};


//class Weight {
//private:
//    map<string, size_t> _map;
//    vector<size_t> _ends;
//    vector<double> _data;
//public:
//    void clear() {
//        _data.clear();
//        _ends.clear();
//        _map.clear();
//    }
//    Weight() {
//    }
//    /**create a dict associate with a file*/
//    Weight(const char* filename) {
//    }
//    void load(const char* filename) {
//        std::ifstream input(filename);
//        string str;
//        vector<double> vec;
//        double v;
//        for (std::string line; std::getline(input, line); ) {
//            std::istringstream iss(line);
//            vec.clear();
//            iss >> str;
//            while (!iss.eof()) {
//                iss >> v;
//                vec.push_back(v);
//            }
//            this->add_from(str, &(vec[0]), vec.size());
//        }
//    }
//    void dump(const char* filename) {
//        std::FILE* pf = fopen(filename, "w");
//        for (auto it = _map.begin();
//                it != _map.end();
//                ++ it) {
//            fprintf(pf, "%s", it->first.c_str());
//            size_t end_ind = it->second;
//            size_t begin = ((end_ind > 0)?_ends[end_ind - 1]:0);
//            size_t end = _ends[end_ind];
//            for (size_t i = begin; i < end; i++) {
//                fprintf(pf, "\t%g", _data[i]);
//            }fprintf(pf, "\n");
//        }
//    }
//
//    void dbg(const string& key) {
//        double* ptr;
//        size_t len;
//        get(key, ptr, len);
//        printf("[%s]", key.c_str());
//        for (size_t i = 0; i < len; i++) {
//            printf(" %g", ptr[i]);
//        }
//        printf("\n");
//    }
//
//    void insert(const string& key, const size_t length){
//        // check existing
//        auto result = _map.find(key);
//        if (result != _map.end()) {
//            return;
//        }
//        // memory
//        _map[key] = _ends.size();
//        for (size_t i = 0; i < length; i++) {
//            _data.push_back(0);
//        }
//        _ends.push_back(_data.size());
//    };
//    void get(const string& key, double*& ptr, size_t& len) {
//        ptr = nullptr;
//        len = 0;
//        auto result = _map.find(key);
//        if (result == _map.end()) {
//            return;
//        }
//        size_t end_off = result->second;
//        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
//        size_t end = _ends[end_off];
//        ptr = &(_data[0]) + begin;
//        len = end - begin;
//    }
//    double* get(const string& key) {
//        auto result = _map.find(key);
//        if (result == _map.end()) {
//            return nullptr;
//        }
//        size_t end_off = result->second;
//        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
//        return &(_data[0]) + begin;
//    }
//    void add_from(const string& key, const double* ptr, 
//            const size_t len, double eta = 1.0) {
//        bool is_all_zero = true;
//        for (size_t i = 0; i < len; i++) {
//            if (ptr[i]) {
//                is_all_zero = false;
//                break;
//            }
//        }
//        if (is_all_zero) return;
//
//        auto result = _map.find(key);
//        if (result == _map.end()) {
//            insert(key, len);
//        }
//
//        double* m = get(key);
//        for (size_t i = 0; i < len; i++) {
//            m[i] += ptr[i] * eta;
//        }
//    }
//    void add_to(const string& key, double* ptr) {
//        auto result = _map.find(key);
//        if (result == _map.end()) {
//            return;
//        }
//        size_t end_off = result->second;
//        size_t begin = (end_off == 0)?0:_ends[end_off - 1];
//        size_t end = _ends[end_off];
//        for (size_t i = begin; i < end; i++) {
//            ptr[i - begin] += _data[i];
//        }
//    }
//    void update(Weight& other, double eta) {
//        for (auto it = other._map.begin();
//                it != other._map.end();
//                ++ it) {
//            double* ptr;
//            size_t len;
//            other.get(it->first, ptr, len);
//            /// if all zero, do nothing
//            /// this is especially helpful for memory-saving
//            if (std::all_of(ptr, ptr+len, [](double x){return x==0;})) {
//                continue;
//            }
//            this->add_from(it->first, ptr, len, eta);
//        }
//    }
//};
//

}


