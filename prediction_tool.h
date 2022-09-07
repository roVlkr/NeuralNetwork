#ifndef PREDICTION_TOOL_H_
#define PREDICTION_TOOL_H_

#include <iostream>
#include "mat.h"
#include "dataconverter.h"
#include "net.h"

using std::cout;
using std::cin;
using std::endl;


class PredictionTool {
public:
  PredictionTool();

  ~PredictionTool();


  /**
   * Trains the net with the given data.
   *
   * @param shuffles - How often the data sets will change.
   * @param training_number - How much data sets / training examples
   *                          there are.
   * @param epochs - How often the net will be trained with the given
   *                 data sets.
   */
  void train(int shuffles, int training_number, int epochs);


  /**
   * Produces an output with the current net configuration, that is,
   * the current weights of the neurons.
   *
   * @param data - The input data to produce an output with.
   * @returns The output of the output neurons produced by the given
   *          input.
   */
  Mat execute(vector<double> const &data);


  /**
   * Wraps the data of the training data file given by the Converter
   * object.
   *
   * @returns The data that is read by Converter.
   */
  inline vector<double> const& getTrainingData() const {
    return converter->getData();
  }


  /**
   * Wraps the input number of the Converter object to identify
   * the data size of the 'known' data, the information to produce a
   * prediction. This is also the number of input neurons of the neural
   * net.
   *
   * @returns The number of input data.
   */
  inline int getInputNumber() const {
    return converter->getInputNumber();
  }


  /**
   * Wraps the output number of the Converter object to identify
   * the data size of the 'unknown' data, the data to produced by the net
   * with a given input.
   * This is also the number of output neurons of the neural net.
   *
   * @returns The number of input data.
   */
  inline int getOutputNumber() const {
    return converter->getOutputNumber();
  }


private:
  Net *net { nullptr };
  DataConverter *converter { nullptr };

  vector<int> initStructure();
  int initType();
  double initSimple();
  backprop_bundle initSA();
  backprop_bundle initRProp();
};

#endif
