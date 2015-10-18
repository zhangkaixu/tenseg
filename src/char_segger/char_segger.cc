#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "char_dict.h"
#include "char_searcher.h"
#include "char_eval.h"

using std::string;
using std::vector;

const size_t N = 4;

void load_corpus(
        const string& filename,
        vector<string>& raws,
        vector<vector<size_t>>& tags
        ){

    std::ifstream input(filename);
    for (std::string line; std::getline(input, line); ) {
        vector<char> word_buffer;
        vector<size_t> tag;
        int cn = 0;
        size_t start = 0;
        size_t i = 0;
        for (; i < line.size(); i++) {
            char c = line.c_str()[i];
            if (c == 32) {
                if (cn == 0) continue;
                if (cn == 1) {
                    tag.push_back(3);
                } else {
                    tag.push_back(0);
                    for (size_t j = 0; j < cn - 2; j++) tag.push_back(1);
                    tag.push_back(2);
                }
                cn = 0;
                start = i + 1;
            } else if ((0xc0 == (c & 0xc0))
                    || !(c & 0x80) ) { // first char
                word_buffer.push_back(c);
                cn ++;
            } else {
                word_buffer.push_back(c);
            }
        }
        if (cn == 1) {
            tag.push_back(3);
        } else if (cn > 1) {
            tag.push_back(0);
            for (size_t j = 0; j < cn - 2; j++) tag.push_back(1);
            tag.push_back(2);
        }
        word_buffer.push_back(0);
        raws.push_back(string(&(word_buffer[0])));
        tags.push_back(tag);
    }
};



void calc_emission(dict::Dict& model, const string& raw,
        vector<double>& emission, bool update) {
    //fprintf(stderr, "%lu\n", raw.size());
    vector<size_t> begins;
    for (size_t i = 0; i < raw.size(); i++) {
        if ((0xc0 == (raw[i] & 0xc0))
                || !(raw[i] & 0x80)) { // first char
            begins.push_back(i);
        }
    }
    begins.push_back(raw.size());

    if (!update) {
        emission.clear();
        for (size_t i = 0; i < N * (begins.size() - 1); i ++) {
            emission.push_back(0);
        }
    }

    int n = 1;
    for (size_t i = 0; i < begins.size() - n; i++) {
        string uni = raw.substr(begins[i], 
                begins[i + n] - begins[i]);

        if (uni[0] == '|') {
            uni = string("，");
        }

        double* m = model.get(uni);
        if (m == nullptr) {
            if (update == false) {
                continue;
            } else {
                model.insert(uni, (n + 2) * N);
                m = model.get(uni);
            }
        };

        for (int j = 0; j < (2 + n) * N; j++) {
            int off = ((int)i - 1) * N + j;
            if (off < 0) continue;
            if (off >= (begins.size() - 1) * N) continue;
            if (update == false) {
                emission[off] += m[j];
            } else {
                m[j] += emission[off];
            }
        }
    }

    n = 2;
    if (begins.size() > n) {
        for (size_t i = 0; i < (int)(begins.size()) - n; i++) {
            string uni = raw.substr(begins[i], 
                    begins[i + n] - begins[i]);

            double* m = model.get(uni);
            if (m == nullptr) {
                if (update == false) {
                    continue;
                } else {
                    model.insert(uni, (n + 2) * N);
                    m = model.get(uni);
                }
            };

            for (int j = 0; j < (2 + n) * N; j++) {
                int off = ((int)i - 1) * N + j;
                if (off < 0) continue;
                if (off >= (begins.size() - 1) * N) continue;
                if (update == false) {
                    emission[off] += m[j];
                } else {
                    m[j] += emission[off];
                }
            }
        }
    }
}

void tagging(dict::Dict& model, string& raw,
        vector<size_t>& tags) {

    tags.clear();
    if (raw.size() == 0) return;

    vector<double> emission;
    vector<double> transition(N * N, 0);
    // cal trans
    model.add_to(string("transition"), &(transition[0]));
    // cal emi
    calc_emission(model, raw, emission, false);
    // viterbi
    tenseg::viterbi(N, transition, emission, tags);
}

void update(dict::Dict& model, string& raw,
        vector<size_t>& result, vector<size_t> gold) {

    vector<double> emission(N * result.size(), 0);
    for (size_t i = 0; i < result.size(); i++) {
        emission[i * N + result[i]] -= 1;
        emission[i * N + gold[i]] += 1;
    }
    calc_emission(model, raw, emission, true);

    vector<double> transition(N * N, 0);
    for (size_t i = 0; i < result.size() - 1; i++) {
        transition[result[i] * N + result[i + 1]] -= 1;
        transition[gold[i] * N + gold[i + 1]] += 1;
    }
    model.add_from(string("transition"), &(transition[0]),
            transition.size());

}


void test(dict::Dict& model, 
        vector<string>& test_raws,
        vector<vector<size_t>>& test_tags
        ) {
    tenseg::Eval e;
    vector<size_t> result;

    for (size_t i = 0; i < test_raws.size(); i++) {
        tagging(model, test_raws[i], result);
        e.eval(result, test_tags[i]);
    }
    e.report();
}

void train(dict::Dict& model, vector<string>& train_raws,
        vector<vector<size_t>>& train_tags,
        vector<string>& test_raws,
        vector<vector<size_t>>& test_tags, size_t iter) {

    dict::Dict ave;
    dict::Dict gradient;
    dict::Learner learner;
    vector<size_t> result;
    for (size_t it = 0; it < iter; it++) {
        printf("it %lu\n", it);
        tenseg::Eval e;
        for (size_t i = 0; i < train_raws.size(); i++) {
            tagging(model, train_raws[i], result);
            e.eval(result, train_tags[i]);

            gradient.clear();
            bool flag_right = true;
            for (size_t j = 0; j < result.size(); j++) {
                if (result[j] != train_tags[i][j]) {
                    flag_right = false;
                    break;
                }
            }
            if (!flag_right) {
                update(gradient, train_raws[i], result, train_tags[i]);
                learner.update(model, gradient);
            }
        }
        e.report();

        e.reset();
        learner.average(model, ave);
        for (size_t i = 0; i < test_raws.size(); i++) {
            tagging(ave, test_raws[i], result);
            e.eval(result, test_tags[i]);
        }
        e.report();
    }
    ave.dump("model.txt");
    model.clear();
    model.update(ave, 1);
}

void do_viterbi(const char* modelfile) {
    dict::Dict model;
    model.load(modelfile);
    std::string cmd;

    std::vector<size_t> tags;
    vector<double> emission;

    vector<double> transition(N * N, 0);
    model.add_to(string("transition"), &(transition[0]));

    for (std::string line; std::getline(std::cin, line); ) {
        emission.clear();
        std::istringstream iss(line);
        double v = 0.0;
        while (!iss.eof()) {
            iss >> v;
            emission.push_back(v);
        }
        tenseg::viterbi(N, transition, emission, tags);
        for (size_t i = 0; i < tags.size(); i++) {
            if (i > 0) printf(" ");
            printf("%lu", tags[i]);
        }printf("\n");
        fflush(stdout);
    }
}

void get_emission(const char* modelfile) {
    dict::Dict model;
    model.load(modelfile);
    std::string cmd;

    vector<double> emission;
    for (std::string line; std::getline(std::cin, line); ) {
        emission.clear();
        calc_emission(model, line, emission, false);
        for (size_t i = 0; i < emission.size(); i++) {
            if (i > 0) printf(" ");
            printf("%g", emission[i]);
        }printf("\n");
        fflush(stdout);
    }
}

void predict(const char* modelfile) {
    dict::Dict model;
    model.load(modelfile);
    std::string cmd;

    std::vector<size_t> tags;
    for (std::string line; std::getline(std::cin, line); ) {
        tagging(model, line, tags);
        size_t char_n = 0;
        for (size_t i = 0; i < line.size(); i++) {
            char& c = line[i];
            if ((0xc0 == (c & 0xc0))
                    || !(c & 0x80) ) {
                if (char_n > 0 && (tags[char_n] == 0 || tags[char_n] == 3)) {
                    printf(" ");
                }
                char_n++;
            }
            printf("%c", c);
        }
        printf("\n");
        fflush(stdout);
    }
}

void shell() {
    // train or test
    dict::Dict model;
    size_t iter = 10;

    vector<string> train_raws;
    vector<vector<size_t>> train_tags;

    vector<string> test_raws;
    vector<vector<size_t>> test_tags;

    std::string cmd;
    for (std::string line; std::getline(std::cin, line); ) {
        std::istringstream iss(line);
        iss >> cmd;
        if (cmd.size() == 0 || (cmd.size() == 1 && cmd[0] == '\n')) continue;
        if (cmd == "quit") {
            fprintf(stderr, "buy~\n");
            break;
        }
        if (cmd == string("training_data")) {
            string filename;
            iss >> filename;
            load_corpus(filename, train_raws, train_tags);
            fprintf(stderr, "load file '%s' as training data\n", filename.c_str());
            continue;
        }
        if (cmd == string("test_data")) {
            string filename;
            iss >> filename;
            load_corpus(filename, test_raws, test_tags);
            fprintf(stderr, "load file '%s' as test data\n", filename.c_str());
            continue;
        }
        if (cmd == string("iteration")) {
            iss >> iter;
            fprintf(stderr, "iteration is set to %lu\n", iter);
            continue;
        }
        if (cmd == string("train")) {
            train(model, train_raws, train_tags, test_raws, test_tags, iter);
            continue;
        }
        if (cmd == string("save")) {
            string filename;
            iss >> filename;
            model.dump(filename.c_str());
            fprintf(stderr, "save model to file '%s'\n", filename.c_str());
            continue;
        }
        if (cmd == string("load")) {
            string filename;
            iss >> filename;
            model.load(filename.c_str());
            fprintf(stderr, "load model from file '%s'\n", filename.c_str());
            continue;
        }
        if (cmd == string("test")) {
            test(model, test_raws, test_tags);
            continue;
        }
    }
}

void print_help_info(const char* argv[]) {
    fprintf(stderr, "A character-based Chinese word segmentor\n");
    fprintf(stderr, "    by Zhang, Kaixu (zhangkaixu@hotmail.com)\n");
    fprintf(stderr, "shell like interface: %s\n", argv[0]);
    fprintf(stderr, "segment by providing a model file: %s modelfile < inputfile > outputfile\n", argv[0]);
}

int main(int argc, const char *argv[])
{
    print_help_info(argv);
    if (argc > 2) {
        if (argv[1][0] == 'v') {
            do_viterbi(argv[2]);
        }
        if (argv[1][0] == 'e') {
            get_emission(argv[2]);
        }
    }
    if (argc > 1) {
        predict(argv[1]);
    }

    shell();

    return 0;
}
