// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> class A {
  T a;

public:
  T get_a() { return a; }
  void set_a(T val) { a = val; }
};

template <class T1> class B : public A<T1> {
public:
  T1 b;
};

int main() {
  B<int> o;
  o.set_a(11);
  o.b = 15;
  printf("%d %d\n", o.get_a(), o.b); // CHECK: 11 15
  A<float> o2;
  o2.set_a(12.34);
  printf("%f\n", o2.get_a()); // CHECK: 12.34
  return 0;
}
