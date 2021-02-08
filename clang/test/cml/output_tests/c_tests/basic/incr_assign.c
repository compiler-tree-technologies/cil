// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int k = 1;
  int a;
  a = k++;
  // CHECK: 1 2
  printf("%d %d\n", a, k);
  a = ++k;
  // CHECK: 3 3
  printf("%d %d\n", a, k);
  return 0;
}
