#pragma once
#include <map>
//#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>

namespace tenseg {
using namespace std;

template<typename V = string>
class Dictionary {
private:
    map<string, V> _dict;
public:
    void load(const char* filename) {
        std::ifstream input(filename);
        string key;
        V value;
        for (std::string line; std::getline(input, line); ) {
            std::istringstream iss(line);
            iss >> key >> value;
            _dict[key] = value;
        }
    }
    bool get(const string& key, V& value) const{
        auto result = _dict.find(key);
        if (result == _dict.end()) {
            return false;
        }
        value = result->second;
        return true;

    }
    bool exists(const string& key) const{
        auto result = _dict.find(key);
        return (result != _dict.end());
    }
};

}
