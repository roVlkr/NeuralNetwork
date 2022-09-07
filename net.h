#ifndef NET_H_
#define NET_H_

#include "helper.h"
#include "pattern.h"
#include "simple_layer.h"
#include "sa_layer.h"
#include "rprop_layer.h"
#include <iostream>
#include <vector>
#include <memory>

using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;


class Net {
public:
  ~Net() { }

  Net& operator=(Net const&) = delete;

  Net& operator=(Net &&net) {
    number_layers_ = net.number_layers_;
    layer_ = std::move(net.layer_);
    training_ = std::move(net.training_);

    net.number_layers_ = 0;

    return *this;
  }

  void setTraining(Pattern &&training) {
    training_ = std::move(training);
  }

  void feedforward(Mat const &input) {
    layer_[0]->reset(input);

    for (auto i { 1 }; i < number_layers_; i++) {
      layer_[i]->reset( layer_[i - 1]->getOutput() );
    }
  }

  void backprop(Mat const &training_out) {
    for (auto i { number_layers_ - 1 }; i >= 0; i--) {
      if (layer_[i]->isHidden())
        layer_[i]->calcWeightChanges( *layer_[i + 1] );
      else // output layer
        layer_[i]->calcWeightChanges(training_out);
    }
  }

  // Ein Trainingsdurchlauf
  // returns error
  double train(Pattern const& training) {
    double error { 0 };

    for (auto t { 0 }; t < training.number(); t++) {
      feedforward(training.input(t));

      // Fehlerberechnung nach Eingabe
      error += (getOutput() - training.output(t)).normF2();

      backprop(training.output(t));
    }

    for (auto i { 0 }; i < number_layers_; i++)
      layer_[i]->applyChanges();

    return error;
  }


  void train(int epochs, bool verbose) {
    for (auto i { 0 }; i < epochs; i++) {
      double error { train(training_) };

      if (verbose) {
        cout << "Datensatz " << training_.name() << ": ";
        cout << "Fehlerwert in Epoche " << i << ": ";
        cout << error << endl;
      }
    }
  }

  Mat const& getOutput() {
    return layer_[number_layers_ - 1]->getOutput();
  }

  Pattern& getTraining() {
    return training_;
  }


  //Factory functions as friend functions
  friend Net* makeSimpleNet(vector<int> const &structure,
    double learning_rate);

  friend Net* makeSANet(vector<int> const &structure,
    backprop_bundle const& bundle);

  friend Net* makeRPropNet(vector<int> const &structure,
    backprop_bundle const& bundle);


private:
  int number_layers_ { 0 }; // Input-Layer wird nicht mitgerechnet
  vector< shared_ptr<Layer> > layer_;

  Pattern training_ { };

  // privater Standardkonstruktor
  Net() { }
};


// Factory functions

Net* makeSimpleNet(vector<int> const &structure, double learning_rate);

Net* makeSANet(vector<int> const &structure, backprop_bundle const& bundle);

Net* makeRPropNet(vector<int> const &structure, backprop_bundle const& bundle);

#endif
