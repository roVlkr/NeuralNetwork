#include "prediction_tool.h"


PredictionTool::PredictionTool() {
  vector<int> structure { initStructure() };
  int type { initType() };

  // Erfasse Daten aus der Datei 'trainingdata'
  converter = new DataConverter("trainingdata",
                                *structure.begin(),
                                *(structure.end() - 1));

  switch (type) {
  case 0:
    cout << "simple" << endl;
    net = makeSimpleNet(structure, initSimple());
    break;
  case 1:
    cout << "selfadaptive" << endl;
    net = makeSANet(structure, initSA());
    break;
  case 2:
    cout << "rprop" << endl;
    net = makeRPropNet(structure, initRProp());
    break;
  default:
    cout << "simple" << endl;
    net = makeSimpleNet(structure, 0.02);
    break;
  }
}


PredictionTool::~PredictionTool() {
  if (net != nullptr)
    delete net;

  if (converter != nullptr)
    delete converter;
}


void PredictionTool::train(int shuffles, int training_number, int epochs) {
  for (auto j { 1 }; j <= shuffles; j++) {
    converter->convert(training_number);

    net->setTraining({
      converter->getInput(),
      converter->getOutput(),
      std::to_string(j)
    });

    net->train(epochs, true);
  }
}


Mat PredictionTool::execute(vector<double> const &data) {
  net->feedforward({ // data as single column matrix
      static_cast<int>(data.size()), 1, data.data()
  });

  return net->getOutput()
              .apply(sigmoid_inv)
              .apply(scale(converter->getScaleFactor()));
}


vector<int> PredictionTool::initStructure() {
  std::string in;
  std::vector<int> structure;
  cout << "Struktur des Netzes: [Zahl / '-' zum Beenden]" << endl;

  for (auto i { 0 }; true; i++) {
    if (i == 0) cout << "Inputlayer: ";
    else cout << "Layer " << i << ": ";

    cin >> in;
    if (in == "-") break;

    structure.push_back(std::stoi(in));
  }

  return structure;
}


int PredictionTool::initType() {
  std::string in;
  cout << "Typ des Netzes [simple = 0 / selfadaptive = 1 / Rprop = 2]: ";
  cin >> in;

  return std::stoi(in);
}


double PredictionTool::initSimple() {
  double learning_rate;

  cout << "Lernrate: ";
  cin >> learning_rate;

  return learning_rate;
}


backprop_bundle PredictionTool::initSA() {
  double init_rate;
  double lower_bound;
  double upper_bound;

  cout << "Initiale Lernrate: ";
  cin >> init_rate;

  cout << "Mindestrate: ";
  cin >> lower_bound;

  cout << "Maximalrate: ";
  cin >> upper_bound;

  return { init_rate, lower_bound, upper_bound };
}


backprop_bundle PredictionTool::initRProp() {
  double init_rate;
  double lower_bound;
  double upper_bound;

  cout << "Initiale Lernrate: ";
  cin >> init_rate;

  cout << "Minimale Schrittweite: ";
  cin >> lower_bound;

  cout << "Maximale Schrittweite: ";
  cin >> upper_bound;

  return { init_rate, lower_bound, upper_bound };
}
