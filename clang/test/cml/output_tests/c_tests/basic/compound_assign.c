#include <stdio.h>

// RUN: %cml %s -o %t && %t | FileCheck %s
int main() {
  int a = 10;
  // CHECK: 10
  printf("%d\n", a);
  a += 10;
  // CHECK: 20
  printf("%d\n", a);
  a -= 10;
  // CHECK: 10
  printf("%d\n", a);
  a *= 10;
  // CHECK: 100
  printf("%d\n", a);
  a /= 10;
  // CHECK: 10
  printf("%d\n", a);
  a %= 10;
  // CHECK: 0
  printf("%d\n", a);
  return 0;
}
