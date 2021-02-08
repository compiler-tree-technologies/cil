// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  double pi = 3.14;
  a = (int)pi;
  // CHECK: 3
  printf("%d\n", a);
  return 0;
}
