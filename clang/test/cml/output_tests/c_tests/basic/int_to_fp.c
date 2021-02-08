// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  double a;
  int b = 10;
  a = (double)b;
  // CHECK: 10.000000
  printf("%lf\n", a);
  return 0;
}
