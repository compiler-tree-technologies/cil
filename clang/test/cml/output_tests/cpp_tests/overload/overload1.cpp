// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>
int func(int val) {
  return val;
}

float func(float val) {
  return val;
}
int main() {
  int a = func(10);
  float val = 10.32;
  float b = func(val);
  a = a +1;
  b = b  + 2.0;
  printf("%d\n", a); // CHECK: 11
  printf("%f\n", b); // CHECK: 12.32
  return 0;
}
