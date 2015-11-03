#pragma once
#include <map>
#include <cmath>
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

    void load(const string& filename) {
        clear();
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
        fprintf(stderr, "load %lu weights\n", _map.size());
    }
    void dump(const string& filename) {
        std::FILE* pf = fopen(filename.c_str(), "w");
        fprintf(stderr, "dump %lu weights\n", _map.size());

        double* ptr;
        size_t len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            fprintf(pf, "%s", it->first.c_str());
            get(it->first, ptr, len);

            for (size_t i = 0; i < len; i++) {
                fprintf(pf, "\t%g", ptr[i]);
            }fprintf(pf, "\n");
        }
        fclose(pf);
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
    void inverse() {
        double* ptr;
        size_t len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            get(it->first, ptr, len);
            for (size_t i = 0; i < len; i++) {
                ptr[i] = 1.0 / ptr[i];
            }
        }
    }
    void power() {
        double* ptr;
        size_t len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            get(it->first, ptr, len);
            for (size_t i = 0; i < len; i++) {
                ptr[i] = ptr[i] * ptr[i];
            }
        }
    }
    void safe_sqrt(double de) {
        double* ptr;
        size_t len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            get(it->first, ptr, len);
            for (size_t i = 0; i < len; i++) {
                if (ptr[i] > 0) {
                    ptr[i] = sqrt(ptr[i]);
                } else {
                    ptr[i] = de;
                }
            }
        }
    }
    void ada_update(Weight& other) {
        double* ptr;
        size_t len;
        double* o_ptr;
        size_t o_len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            get(it->first, ptr, len);
            other.get(it->first, o_ptr, o_len);
            if (!o_ptr || o_len != len) {
                for (size_t i = 0; i < len; i++) {
                    ptr[i] = ptr[i];
                }
            } else {
                for (size_t i = 0; i < len; i++) {
                    if (o_ptr[i] > 0) {
                        ptr[i] /= sqrt(o_ptr[i]);
                    } else {
                        ptr[i] /= 1;
                    }
                }
            }

        }
    }

    void multiply(Weight& other) {
        double* ptr;
        size_t len;
        double* o_ptr;
        size_t o_len;
        for (auto it = _map.begin();
                it != _map.end();
                ++ it) {
            get(it->first, ptr, len);
            other.get(it->first, o_ptr, o_len);
            if (!o_ptr || o_len != len) {
                for (size_t i = 0; i < len; i++) {
                    ptr[i] = 0;
                }
                
                continue;
            }

            for (size_t i = 0; i < len; i++) {
                ptr[i] *= o_ptr[i];
            }
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

class IWeight {
public:
    /// access
    virtual void get(const string& key, double*& ptr, size_t& len) = 0;
    virtual double* get(const string& key) = 0;
    virtual void add_to(const string& key, double* ptr) = 0;

    /// modify
    virtual void clear() = 0;
    virtual void insert(const string& key, const size_t length) = 0;
    virtual void add_from(const string& key, const double* ptr, 
            const size_t len, double eta = 1.0) = 0;
    virtual void update(IWeight& other, double eta) = 0;

    /// filesystem
    virtual void load(const char* filename) = 0;
    virtual void dump(const char* filename) = 0;
};

}


