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

}
