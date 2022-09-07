#ifndef RPROP_LAYER_H_
#define RPROP_LAYER_H_

#include "layer.h"

class RProp_Layer : public Layer {
public:
  RProp_Layer() { }

  RProp_Layer(int last_length, int this_length, bool hidden,
           backprop_bundle const &bundle) :
    Layer(last_length, this_length, hidden),
    bundle_ { bundle },
    gradient_ { fillMat(this_length, last_length + 1, 0) }
  { }


  // Output-Layer-Variante
  void calcWeightChanges(Mat const &training_output) override {
    Mat out_derivative { out_.apply(sig_deriv) };
    Mat error { training_output - out_ };

    delta_ = std::move(error) ^ out_derivative;
    gradient_ += delta_ * in_.transposed();
  }


  // Hidden-Layer-Variante
  void calcWeightChanges(Layer const &next) override {
    calcDelta(next);

    gradient_ += delta_ * in_.transposed();
  }


  void applyChanges() override {
    rpropChanges(bundle_);

    weights_ += weights_diff_ ^ gradient_.apply(sgn);

    gradient_old_ = gradient_;      // Speichere alten Wert
    gradient_.setAll(0);            // "= 0" zum erneuten Aufsummieren
  }


private:
  Mat gradient_;
  Mat gradient_old_;
  std::vector<bool> shrinked_;

  backprop_bundle bundle_;


  void rpropChanges(backprop_bundle const &b) {
    if (gradient_old_.getSize() == 0) { // Noch keine 2 Durchläufe gehabt
      weights_diff_ = fillMat(weights_.getRows(), weights_.getCols(), b.init_rate_);
      shrinked_ = std::vector<bool>(weights_.getSize(), false);
      return;
    }

    for (auto w { 0 }; w < weights_.getSize(); w++) {
      auto cond { gradient_.get(w) * gradient_old_.get(w) };
      auto dw { weights_diff_.get(w) };

      if (cond < 0) {
        weights_diff_.set(w, std::max(dw * b.shrink_, b.lower_bound_));
        shrinked_[w] = true;
      } else if (cond > 0 && !shrinked_[w]) {
        weights_diff_.set(w, std::min(dw * b.grow_, b.upper_bound_));
        shrinked_[w] = false;
      } else
        shrinked_[w] = false;
    }
  }
};

#endif
