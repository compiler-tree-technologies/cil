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
  return 0;
}
