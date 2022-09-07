#ifndef MAT_H_
#define MAT_H_

#include <initializer_list>
#include <functional>
#include <iostream>


class Mat {
public:
  Mat();
  Mat(int rows, int cols, double *data = nullptr);
  Mat(int rows, int cols, double const *data);
  Mat(std::initializer_list< std::initializer_list<double> > l);
  Mat(Mat const &mat);
  Mat(Mat &&mat);
  ~Mat();

  Mat& operator=(Mat const &mat);
  Mat& operator=(Mat &&mat) noexcept;

  // Member functions /////////////////////////////////////
  // Getter
  inline int getSize() const { return size; }
  inline int getRows() const { return rows; }
  inline int getCols() const { return cols; }

  // Zugriff auf Element über [zeile][spalte]
  inline double* operator[](int row) { return data + row * cols; }

  // "const-Variante" des []-Operators
  inline double get(int row, int column) const {
    return data[row * cols + column];
  }

  // Set all entries equal to d
  void setAll(double d);

  // Coordinatewise application of the function f
  Mat apply(std::function<double(double)> f) const;

  // This matrix in transposed form
  Mat transposed() const;

  // Squared Frobenius norm (with vectors: squared Euclidian norm)
  double normF2() const;

  // Scalar multiplication
  Mat& operator*=(double k) {
    for (auto i { 0 }; i < size; i++) data[i] *= k;
    return *this;
  }

  // Coordinatewise multiplication ( {1, 2} ^ {2, 3} = {2, 6} )
  Mat& operator^=(Mat const &m);

  // Matrix addition
  Mat& operator+=(Mat const &m);

  // Matrix subtraction
  Mat& operator-=(Mat const &m);

  // Matrix multiplication
  Mat operator*(Mat const &m) const;

  // This matrix negated
  Mat operator-() const;
  /////////////////////////////////////////////////////////


  // Matrix as Vector /////////////////////////////////////
  Mat(std::initializer_list<double> l);

  // Get the i'th element of the vector
  inline double get(int i) const   { return data[i]; }

  // Set the i'th element of the vector equal to v
  inline void set(int i, double v) { data[i] = v;    }

  // Concatenate the number represented by d to the vector
  Mat concat(double d) const;
  /////////////////////////////////////////////////////////


  // Friend-Funktionen ////////////////////////////////////
  friend Mat operator-(Mat const &m1, Mat &&m2);
  friend std::ostream& operator<< (std::ostream& os, Mat const &m);
  friend Mat fillMat(int, int, double);
  friend Mat fillMat(int, int, double, double);
  friend Mat unitMat(int, int);
  /////////////////////////////////////////////////////////


private:
  double *data { nullptr };
  int rows { };
  int cols { };
  int size { };
};


// Ausgabe einer Matrix über einen Ausgabestrom
std::ostream& operator<<(std::ostream& os, Mat const &m);

// Skalare Multiplikation
inline Mat operator*(double k, Mat m) { m *= k; return m; }
inline Mat operator*(Mat m, double k) { m *= k; return m; }

// Koordinatenweise Multiplikation von Matrizen / Vektoren
inline Mat operator^(Mat m1, Mat const &m2) { m1 ^= m2; return m1; }

// Herkömmliche Addition / Subtraktion von Matrizen
inline Mat operator+(Mat m1, Mat const &m2) { m1 += m2; return m1; }
inline Mat operator-(Mat m1, Mat const &m2) { m1 -= m2; return m1; }
Mat operator-(Mat const &m1, Mat &&m2);

// Factory-Funktion für Matrizen mit gleichen Einträgen
Mat fillMat(int rows, int cols, double d);

Mat fillMat(int rows, int cols, double min, double max);

Mat unitMat(int rows, int cols);

#endif
