// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = -11;
  float f = 11.0;
  // CHECK: -11
  printf("%d\n", a);
  a = -a;
  // CHECK: 11
  printf("%d\n", a);
  // CHECK: 11.000000
  printf("%f\n", f);
  f = -f;
  // CHECK: -11.000000
  printf("%f\n", f);
  return 0;
}
