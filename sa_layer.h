#ifndef SA_LAYER_H_
#define SA_LAYER_H_

#include "layer.h"
#include <vector>


class SA_Layer : public Layer {
public:
  SA_Layer() { };

  SA_Layer(int last_length, int this_length, bool hidden,
           backprop_bundle const &bundle) :
    Layer(last_length, this_length, hidden),
    bundle_ { bundle }
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
    changeLearningRates(bundle_);

    weights_ += learning_rates_ ^ weights_diff_;

    weights_diff_old_ = weights_diff_;  // Speichere alten Wert
    weights_diff_.setAll(0);            // "= 0" zum erneuten Aufsummieren
  }


private:
  Mat learning_rates_;
  Mat weights_diff_old_;
  std::vector<bool> shrinked_;

  backprop_bundle bundle_;


  /////////////////////
  // Hilfsfunktionen //
  /////////////////////

  // Selbstadaptive Backpropagation
  void changeLearningRates(backprop_bundle const &b) {
    if (weights_diff_old_.getSize() == 0) { // Noch keine 2 Durchläufe gehabt
      learning_rates_ =
        fillMat(weights_.getRows(), weights_.getCols(), b.init_rate_);
      shrinked_ = std::vector<bool>(weights_.getSize(), false);

      return;
    }

    for (auto w { 0 }; w < weights_.getSize(); w++) {
      auto cond { weights_diff_.get(w) * weights_diff_old_.get(w) };
      auto rate { learning_rates_.get(w) };

      if (cond < 0) {
        learning_rates_.set(w, std::max(rate * b.shrink_, b.lower_bound_));
        shrinked_[w] = true;
      } else if (cond > 0 && !shrinked_[w]) {
        learning_rates_.set(w, std::min(rate * b.grow_, b.upper_bound_));
        shrinked_[w] = false;
      } else
        shrinked_[w] = false;
    }
  }
};

#endif
