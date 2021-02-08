// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> class A {
public:
  T a;
};

int main() {
  A<int> o;
  A<float> o2;
  o.a = 11;
  o2.a = 13.2;
  printf("%d\n", o.a);  // CHECK: 11
  printf("%f\n", o2.a); // CHECK: 13.2
  return 0;
}
