#include "net.h"


Net* makeSimpleNet(vector<int> const &structure, double learning_rate) {
  auto *net { new Net() };
  net->number_layers_ = static_cast<int>(structure.size() - 1);

  // init layers
  for (auto it { structure.begin() }; it != structure.end() - 1; it++) {
      bool hidden { it != structure.end() - 2 };

      net->layer_.push_back(std::make_shared<Simple_Layer>(
        *it, *(it + 1), hidden, learning_rate
      ));
  }

  return net;
}


Net* makeSANet(vector<int> const &structure,
               backprop_bundle const& bundle)
{
  auto *net { new Net() };
  net->number_layers_ = static_cast<int>(structure.size() - 1);

  for (auto it { structure.begin() }; it != structure.end() - 1; it++) {
      bool hidden { it != structure.end() - 2 };

      net->layer_.push_back(std::make_shared<SA_Layer>(
        *it, *(it + 1), hidden, bundle
      ));
  }

  return net;
}


Net* makeRPropNet(vector<int> const &structure,
                  backprop_bundle const& bundle)
{
  auto *net { new Net() };
  net->number_layers_ = static_cast<int>(structure.size() - 1);

  for (auto it { structure.begin() }; it != structure.end() - 1; it++) {
      bool hidden { it != structure.end() - 2 };

      net->layer_.push_back(std::make_shared<RProp_Layer>(
        *it, *(it + 1), hidden, bundle
      ));
  }

  return net;
}
