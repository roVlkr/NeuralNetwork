#include "layer.h"


Layer::Layer() { };


Layer::Layer(int last_length, int this_length, bool hidden) :
  length_ { this_length },
  hidden_ { hidden }
{
  fillWeights(last_length, this_length);
}


void Layer::reset(Mat const &last_out) {
  in_ = last_out.concat(1);
  out_ = (weights_ * in_).apply(sigmoid);
}


void Layer::fillWeights(int last_length, int this_length) {
  weights_ = fillMat(this_length, last_length + 1, -0.001, 0.001);
  weights_diff_ = fillMat(this_length, last_length + 1, 0);
}


void Layer::calcDelta(Layer const &next) {
  delta_ = Mat(length_, 1); // Für jedes Neuron ein Delta (Spaltenvektor)

  // Für jedes Neuron in diesem Layer
  for (auto u { 0 }; u < length_; u++) {
    double sum { 0 };

    // Summiere über die Neuronen der nachfolgenden Schicht
    for (auto succ { 0 }; succ < next.length_; succ++)
      sum += next.weights_.get(succ, u) * next.delta_.get(succ);

    delta_.set(u, sum * sig_deriv(out_.get(u)));
  }
}
