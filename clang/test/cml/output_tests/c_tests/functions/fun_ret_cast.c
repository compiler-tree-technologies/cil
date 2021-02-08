// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int greater(int a, int b) { return (a > b || b > a); }

int main() {
  // CHECK: 1
  printf("%d\n", greater(10, 5) ? 1 : 0);
  return 0;
}
