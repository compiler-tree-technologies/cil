// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a, b, c;
  b = 10;
  c = 20;
  a = b, c;
  // CHECK: 10
  printf("%d\n", a);
  return 0;
}
