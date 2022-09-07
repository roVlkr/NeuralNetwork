#ifndef HELPER_H_
#define HELPER_H_

#include <cmath>
#include <limits>
#include <functional>

// Skalierungsfaktor, sodass sigmoid einen Wertebereich (-alpha, alpha) hat
constexpr double alpha { 1 }; // > 1

// Die Sigmoidfunktion und Verwandte
double sigmoid(double in);
double sig_deriv(double sig);
double sigmoid_inv(double d);

// Die Signum-Funktion
double sgn(double d);

std::function<double(double)> scale(double k);

// Datenbündel für selbstadaptive Backpropagation
struct backprop_bundle {
  double init_rate_ { 0 };
  double shrink_ { 0 };
  double grow_ { 0 };
  double lower_bound_ { 0 };
  double upper_bound_ { 0 };

  backprop_bundle() { }

  backprop_bundle(double init_rate,
      double shrink,
      double grow,
      double lower_bound = std::numeric_limits<double>::min(),
      double upper_bound = std::numeric_limits<double>::max()) :
    init_rate_ { init_rate },
    shrink_ { shrink },
    grow_ { grow },
    lower_bound_ { lower_bound },
    upper_bound_ { upper_bound }
  { }
};

#endif
