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

int main() {
  B<int, float> o;
  o.set_a(11);
  o.b = 15;
  o.set_c(12.34);
  printf("%d %d %f\n", o.get_a(), o.b, o.get_c()); // CHECK: 12.34
  return 0;
}
