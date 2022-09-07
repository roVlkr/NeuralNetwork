#ifndef LAYER_H_
#define LAYER_H_

#include "mat.h"
#include "helper.h"


// Inputlayer wird nicht hierunter gefasst
class Layer {
public:
  Layer();
  Layer(int last_length, int this_length, bool hidden);

  void reset(Mat const &last_out);

  // Output-Layer-Variante
  virtual void calcWeightChanges(Mat const &training_output) = 0;

  // Hidden-Layer-Variante
  virtual void calcWeightChanges(Layer const &next) = 0;

  virtual void applyChanges() = 0;

  inline int getLength() const { return length_; }
  inline bool isHidden() const { return hidden_; }
  inline Mat const& getInput() const { return in_; }
  inline Mat const& getOutput() const { return out_; }
  inline Mat const& getDelta() const { return delta_; }
  inline Mat const& getWeights() const { return weights_; }

protected:
  Mat in_;
  Mat out_;
  Mat delta_;
  Mat weights_;
  Mat weights_diff_;

  int length_ { 0 };
  bool hidden_ { true };


  /////////////////////
  // Hilfsfunktionen //
  /////////////////////

  // init-Funktion für weights-Matrix
  void fillWeights(int last_length, int this_length);

  // Hidden-Layer-Variante
  void calcDelta(Layer const &next);
};

#endif
