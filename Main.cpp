#include <iostream>
#include "prediction_tool.h"

using std::cout;
using std::endl;

/*
  Windows-Eintrag!
*/


int main() {
  PredictionTool tool;
  tool.train(10, 50, 500);

  auto first = tool.getTrainingData().end() - tool.getInputNumber();
  vector<double> data { first, tool.getTrainingData().end() };

  cout << tool.execute(data) << endl;

  return 0;
}
