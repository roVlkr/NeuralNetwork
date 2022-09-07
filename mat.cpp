#include "mat.h"
#include <stdexcept>
#include <random>
#include <iostream>


//////////////////////////////////////////////////////
// Konstruktoren, Destruktor, Assignment-Operatoren //
//////////////////////////////////////////////////////

Mat::Mat() {
  rows = 0;
  cols = 0;
  size = 0;
  data = nullptr;
}


Mat::Mat(int rows, int cols, double *data) {
  this->rows = rows;
  this->cols = cols;
  this->size = rows * cols;
  this->data = new double[ this->size ];

  if (data != nullptr) {
    for (auto i { 0 }; i < this->size; i++)
       this->data[i] = data[i];
  }
}


Mat::Mat(int rows, int cols, double const *data) {
  this->rows = rows;
  this->cols = cols;
  this->size = rows * cols;
  this->data = new double[ this->size ];

  if (data != nullptr) {
    for (auto i { 0 }; i < this->size; i++)
       this->data[i] = data[i];
  }
}


Mat::Mat(std::initializer_list< std::initializer_list<double> > l) {
  rows = l.size();
  cols = l.begin()->size();
  size = rows * cols;
  data = new double[ size ];

  int i { 0 };
  for (auto row : l)
    for (auto d : row) {
      data[i] = d;
      i++;
    }
}


Mat::Mat(Mat const &mat) {
  rows = mat.rows;
  cols = mat.cols;
  size = mat.size;

  data = new double[ size ];

  for (auto i { 0 }; i < size; i++)
    data[i] = mat.data[i];
}


Mat::Mat(Mat &&mat) {
  rows = mat.rows;
  cols = mat.cols;
  size = mat.size;

  data = mat.data;

  mat.data = nullptr;
  mat.rows = mat.cols = mat.size = 0;
}


Mat::~Mat() {
  if (data != nullptr)
    delete[] data;
}


Mat& Mat::operator=(Mat const &mat) {
  if (this == &mat) return *this;
  if (data != nullptr) delete[] data;

  rows = mat.rows;
  cols = mat.cols;
  size = mat.size;

  data = new double[ size ];

  for (auto i { 0 }; i < size; i++)
    data[i] = mat.data[i];

  return *this;
}


Mat& Mat::operator=(Mat &&mat) noexcept {
  if (this == &mat) return *this;
  if (data != nullptr) delete[] data;

  rows = mat.rows;
  cols = mat.cols;
  size = mat.size;

  data = mat.data;

  mat.data = nullptr;
  mat.rows = mat.cols = mat.size = 0;

  return *this;
}


///////////////////////
// Member-Funktionen //
///////////////////////

Mat Mat::apply(std::function<double(double)> f) const {
  Mat res { *this };

  for (auto i { 0 }; i < size; i++)
    res.data[i] = f(res.data[i]);

  return res;
}


Mat Mat::transposed() const {
  Mat res(cols, rows);

  for (auto i { 0 }; i < rows; i++)
    for (auto j { 0 }; j < cols; j++)
      res[j][i] = get(i, j);

  return res;
}


double Mat::normF2() const {
  double sum { 0 };

  for (auto i { 0 }; i < size; i++)
    sum += data[i] * data[i];

  return sum;
}

void Mat::setAll(double d) {
  for (auto i { 0 }; i < size; i++)
    data[i] = d;
}


Mat Mat::operator-() const {
  Mat res { *this };

  for (auto i { 0 }; i < size; i++)
    res.data[i] = -res.data[i];

  return res;
}


Mat Mat::operator*(Mat const &m) const {
  if (cols != m.rows) {
    throw(std::domain_error("Matrix multiplication failed!"));
  }

  Mat res(rows, m.cols);

  for (auto i { 0 }; i < rows; i++) {
    for (auto j { 0 }; j < m.cols; j++) {

      double sum { 0 };
      for (auto k { 0 }; k < m.rows; k++)
        sum += get(i, k) * m.get(k, j);

      res[i][j] = sum;
    }
  }

  return res;
}


Mat& Mat::operator^=(Mat const &m) {
  if (size != m.size) {
    throw(std::domain_error("Matrix coordinatewise multiplication failed!"));
  }

  for (auto i { 0 }; i < size; i++)
    data[i] *= m.get(i);

  return *this;
}


Mat& Mat::operator+=(Mat const &m) {
  if (rows != m.rows || cols != m.cols) {
    throw(std::domain_error("Matrix addition failed!"));
  }

  for (auto i { 0 }; i < rows; i++)
    for (auto j { 0 }; j < cols; j++)
      (*this)[i][j] += m.get(i, j);

  return *this;
}


Mat& Mat::operator-=(Mat const &m) {
  if (rows != m.rows || cols != m.cols) {
    throw(std::domain_error("Matrix subtraction failed!"));
  }

  for (auto i { 0 }; i < rows; i++)
    for (auto j { 0 }; j < cols; j++)
      (*this)[i][j] -= m.get(i, j);

  return *this;
}


/////////////////////////////////////////////////////////
// Globale Operatoren und operator-ähnliche Funktionen //
/////////////////////////////////////////////////////////

Mat operator-(Mat const &m1, Mat &&m2) {
  if (m1.rows != m2.rows || m1.cols != m2.cols) {
    throw(std::domain_error("Matrix subtraction failed!"));
  }

  Mat res { std::move(m2) };

  for (auto i { 0 }; i < m1.rows; i++)
    for (auto j { 0 }; j < m1.cols; j++)
      res[i][j] = m1.get(i, j) - res[i][j];

  return res;
}


std::ostream& operator<<(std::ostream& os, Mat const &m) {
  for (auto i { 0 }; i < m.rows; i++) {
    for (auto j { 0 }; j < m.cols; j++)
      if (j < m.cols - 1)
        os << m.get(i, j) << ", ";
      else if (i < m.rows - 1)
        os << m.get(i, j) << "\n";
      else
        os << m.get(i, j);
  }

  return os;
}


/////////////////////////////////
// Factory-Funktionen (global) //
/////////////////////////////////

Mat fillMat(int rows, int cols, double value) {
  Mat z(rows, cols);

  for (auto i { 0 }; i < z.size; i++)
    z.data[i] = value;

  return z;
}


Mat unitMat(int rows, int cols) {
  Mat z { fillMat(rows, cols, 0) };

  auto rang { std::min(rows, cols) };
  for (auto i { 0 }; i < rang; i++)
    z[i][i] = 1;

  return z;
}


// Fill with random values in range (min, max).
Mat fillMat(int rows, int cols, double min, double max) {
  Mat z(rows, cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min, max);

  for (auto i { 0 }; i < z.size; i++)
    z.data[i] = dist(gen);

  return z;
}


//////////////////////////////////
// Abschnitt: Matrix als Vektor //
//////////////////////////////////

Mat::Mat(std::initializer_list<double> l) {
  rows = l.size();
  cols = 1;
  size = rows;
  data = new double[ size ];

  int i { 0 };
  for (auto d : l) {
    data[i] = d;
    i++;
  }
}

// Wird hinten drangehängt
Mat Mat::concat(double d) const {
  double *data_new = new double[ size + 1 ];

  for (auto i { 0 }; i < size; i++)
    data_new[i] = data[i];

  data_new[ size ] = d;

  return Mat(rows + 1, 1, data_new);
}
