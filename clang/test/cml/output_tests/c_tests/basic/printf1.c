// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  float a;
  a = 31.0;
  printf("%f\n",a);
  return 0;
}

// CHECK: 31.00 
