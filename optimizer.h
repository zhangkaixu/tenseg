#pragma once
#include "weight.h"

namespace tenseg {

class Learner {
private:
    Weight _acc;
    size_t _step;
public:
    Learner() {
        _step = 0;
    }
    void update(Weight& model, Weight& gradient) {
        _step++;
        model.update(gradient, 1.0);
        _acc.update(gradient, _step);
    }
    void average(Weight& model, Weight& ave) {
        ave.clear();
        ave.update(model, 1.0);
        ave.update(_acc, - 1.0 / _step);
    }
};

}
