// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> class A {
  T a;

public:
  T get_a() { return a; }
  void set_a(T val) { a = val; }
};

template <class T> class C {
  T c;

public:
  T get_c() { return c; }
  void set_c(T val) { c = val; }
};

template <class T1, class T2> class B : public A<T1>, public C<T2> {
public:
  T1 b;
};

template <class T1, class T2, class T3> class D : public B<T1, T2> {
public:
  T3 d;
};

int main() {
  D<int, float, double> o;
  o.set_a(11);
  o.b = 15;
  o.set_c(12.34);
  o.d = 13.96;
  printf("%d %d %f\n", o.get_a(), o.b, o.get_c()); // CHECK: 12.34
  printf("%lf\n",o.d); // CHECK: 13.96
  return 0;
}
