#include "helper.h"


double sigmoid(double in) {
  return alpha / (1 + std::exp(-in));
}

double sig_deriv(double sig) {
  return sig * (1 - sig / alpha);
}

double sigmoid_inv(double d) {
  return std::log(d) - std::log(alpha - d);
}

double sgn(double d) {
  if (d > 0) return 1;
  else if (d == 0) return 0;
  else return -1;
}

std::function<double(double)> scale(double k) {
  return [k] (double d) { return d * k; };
}
