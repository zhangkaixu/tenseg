#pragma once
#include "weight.h"

namespace tenseg {

/**
 * 一个平均感知器的学习类
 * */

template<class Weight>
class Learner {
private:
    Weight _weight;
    Weight _acc;
    size_t _step;
public:
    Learner() {
        _step = 0;
    }
    Weight& weight() {
        return _weight;
    }
    void update(Weight& gradient) {
        _step++;
        _weight.update(gradient, 1.0);
        _acc.update(gradient, _step);
    }
    void average(Weight& ave) {
        ave.clear();
        ave.update(_weight, 1.0);
        ave.update(_acc, - 1.0 / _step);
    }
};

template<class Weight>
class AvgAdaGrad {
private:
    Weight _weight;
    Weight _acc;
    Weight _ss;
    size_t _step;
public:
    AvgAdaGrad() {
        _step = 0;
    }
    Weight& weight() {
        return _weight;
    }
    void update(Weight& gradient) {
        _step++;
        Weight tmp;
        Weight tmp2;
        tmp.update(gradient, 1.0);
        tmp.power();
        _ss.update(tmp, 1.0);


        tmp2.update(gradient, 1.0);
        tmp2.ada_update(_ss);
        _weight.update(tmp2, 1.0);
        _acc.update(tmp2, _step);
    }
    void average(Weight& ave) {
        ave.clear();
        ave.update(_weight, 1.0);
        ave.update(_acc, - 1.0 / _step);
    }
};

}
