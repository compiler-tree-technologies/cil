// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> class A {
  T a;

public:
  T get_a() { return a; }
  void set_a(T val) { a = val; }
};

int main() {
  A<int> o;
  o.set_a(11);
  printf("%d\n", o.get_a()); // CHECK: 11
  A<float> o2;
  o2.set_a(12.34);
  printf("%f\n", o2.get_a()); // CHECK: 12.34
  return 0;
}
