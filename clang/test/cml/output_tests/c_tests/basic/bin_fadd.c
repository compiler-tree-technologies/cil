// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  float a, b;
  a = 31.0;
  b = 20.0;
  a = a + b;
  printf("%f\n",a);
  return 0;
}
// CHECK: 51.0 
