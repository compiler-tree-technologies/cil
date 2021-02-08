// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  float a, b;
  a = 40.0;
  b = 20.0;
  float add = a + b;
  float sub = a - b;
  float mul = a * b;
  float div = a / b;
  printf("%f %f %f %f\n",add,sub,mul,div);
  return 0;
}
// CHECK: 60.000000
// CHECK: 20.000000
// CHECK: 800.000000
// CHECK: 2.000000
