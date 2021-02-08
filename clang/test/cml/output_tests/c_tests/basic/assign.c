// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  int b;
  a = b = 4;
  // CHECK: 4 4
  printf("%d %d\n", a, b);
  return 0;
}
