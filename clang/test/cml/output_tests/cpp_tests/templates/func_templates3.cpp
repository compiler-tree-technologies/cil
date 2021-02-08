// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

template <class T> void add_one(T *val) { *val = *val + 1; }

template <class T1, class T2> void add_two(T1 val1, T2 val2) {
  add_one(val1);
  add_one(val2);
}

int main() {
  int a = 10;
  float b = 11.2;
  add_two(&a, &b);
  printf("%d\n", a); // CHECK: 11
  printf("%f\n", b); // CHECK: 12.2
  return 0;
}
