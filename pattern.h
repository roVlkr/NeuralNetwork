#ifndef PATTERN_H_
#define PATTERN_H_

#include "mat.h"
#include <vector>
#include <string>
#include <iostream>

using std::vector;

class Pattern {
public:
  Pattern() { }

  Pattern(vector<Mat> const &input,
          vector<Mat> const &output,
          std::string name) :
    input_ { input },
    output_ {  output },
    number_ { static_cast<int>(input_.size()) }, // = output_.size();
    name_ { name }
  { }

  Pattern(vector<Mat> &&input,
          vector<Mat> &&output,
          std::string name) :
    input_ { std::move(input) },
    output_ { std::move(output) },
    number_ { static_cast<int>(input_.size()) }, // = output_.size();
    name_ { name }
  { }

  Pattern(Pattern const &p) {
    number_ = p.number_;
    input_ = p.input_;
    output_ = p.output_;
    name_ = p.name_;
  }

  Pattern(Pattern &&p) {
    number_ = p.number_;
    input_ = std::move(p.input_);
    output_ = std::move(p.output_);
    name_ = std::move(p.name_);
  }

  Pattern& operator=(Pattern const &p) = delete;

  Pattern& operator=(Pattern &&p) noexcept {
    if (this == &p) return *this;

    number_ = p.number_;
    input_ = std::move(p.input_);
    output_ = std::move(p.output_);
    name_ = std::move(p.name_);

    return *this;
  }

  Mat const& input(int i) const {
    return input_[i];
  }

  Mat const& output(int i) const {
    return output_[i];
  }

  int number() const {
    return number_;
  }

  std::string name() const {
    return name_;
  }

private:
  vector<Mat> input_;
  vector<Mat> output_;
  int number_ { 0 };
  std::string name_;
};

#endif
