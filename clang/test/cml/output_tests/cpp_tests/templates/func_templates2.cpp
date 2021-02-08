// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> T add_one(T val) {
  val = val + 1;
  return val;
}

template <class T> T add_two(T val) {
  T temp = add_one(val);
  return add_one(temp);
}
int main() {
  int a = add_one(10);
  float b = 11.2;
  b = add_two(b);
  printf("%d\n", a); // CHECK: 11
  printf("%f\n", b); // CHECK: 13.2
  return 0;
}
