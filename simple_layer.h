#ifndef SIMPLE_LAYER_H_
#define SIMPLE_LAYER_H_

#include "layer.h"

class Simple_Layer : public Layer {
public:
  Simple_Layer() { }

  Simple_Layer(int last_length, int this_length, bool hidden,
               double learning_rate) :
    Layer(last_length, this_length, hidden),
    learning_rate_ { learning_rate }
  { }

    // Output-Layer-Variante
    void calcWeightChanges(Mat const &training_output) override {
      Mat out_derivative { out_.apply(sig_deriv) };
      Mat error { training_output - out_ };

      delta_ = std::move(error) ^ out_derivative;
      weights_diff_ += delta_ * in_.transposed();
    }


    // Hidden-Layer-Variante
    void calcWeightChanges(Layer const &next) override {
      calcDelta(next);

      weights_diff_ += delta_ * in_.transposed();
    }


    void applyChanges() override {
      weights_ += learning_rate_ * weights_diff_;

      weights_diff_.setAll(0);
    }

private:
  double learning_rate_;
};

#endif
