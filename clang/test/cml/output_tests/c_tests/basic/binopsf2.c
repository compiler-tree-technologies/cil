// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  float a, b;
  a = 40.0;
  b = 20.0;
  float add = (((a + b) * 10.0) - 2.0);
  printf("%f\n",add);
  return 0;
}
// CHECK: 598
