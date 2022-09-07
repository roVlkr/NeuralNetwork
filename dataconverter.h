#ifndef DATA_CONVERTER_H_
#define DATA_CONVERTER_H_

#include <fstream>
#include <string>
#include <vector>
#include <random>

#include "mat.h"
#include "helper.h"

using std::string;
using std::ifstream;
using std::vector;


class DataConverter {
public:
  DataConverter(string file_name, int input_number, int output_number) :
    file_ { ifstream(file_name) },  // open file
    input_number_ { input_number },
    output_number_ { output_number }
  {
    readFile();
  }

  ~DataConverter() {
    if (file_.is_open())
      file_.close();
  }

  // no const
  inline vector<Mat>&& getInput() {
    return std::forward< vector<Mat> >(input_);
  }

  inline vector<Mat>&& getOutput() {
    return std::forward< vector<Mat> >(output_);
  }

  inline int getInputNumber() const {
    return input_number_;
  }

  inline int getOutputNumber() const {
    return output_number_;
  }

  inline vector<double> const& getData() const {
    return data_;
  }

  inline double getScaleFactor() const {
    return scale_factor_;
  }

  void convert(int training_number) {
    input_ = vector<Mat>(training_number);
    output_ = vector<Mat>(training_number);

    std::random_device rd;
    std::mt19937 gen(rd());

    auto endindex { data_.size() - (input_number_ + output_number_) };
    std::uniform_int_distribution<> dist(
        endindex - 2 * training_number, endindex);

    for (int t { 0 }; t < training_number; t++) {
      int starting_point { dist(gen) };
      input_[t] = Mat(input_number_, 1);
      output_[t] = Mat(output_number_, 1);

      for (int i { 0 }; i < input_number_ + output_number_; i++)
        if (i < input_number_)
          input_[t].set(i, data_[starting_point + i]);
        else
          output_[t].set(i - input_number_, data_[starting_point + i]);

      // Zum besseren Abgleich mit der Ausgabe des Netzes
      output_[t] = output_[t].apply(sigmoid);
    }
  }

private:
  ifstream file_;
  vector<double> data_;
  vector<Mat> input_;
  vector<Mat> output_;
  int const input_number_;
  int const output_number_;
  double scale_factor_ { 1 };


  void readFile() {
    double max { 0 };
    string number;

    if (file_.is_open()) {
      double d { };
      while (std::getline(file_, number)) {
        d = std::stod(number);
        max = std::max(d, max);

        data_.push_back(d);
      }
    }

    // smallest potency of 10 that is greater than max
    this->scale_factor_ = std::pow(10, static_cast<int>(std::log10(max)) + 1);

    // scale the data
    for (auto &d : data_)
      d /= scale_factor_;
  }
};

#endif
