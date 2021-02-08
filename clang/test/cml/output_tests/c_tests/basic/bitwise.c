// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int m = 2;
  int n = 5;
  // CHECK: 2
  printf("%d\n", m);
  m = m << 1;
  // CHECK: 4
  printf("%d\n", m);
  m = m >> 1;
  // CHECK: 2
  printf("%d\n", m);
  n = m | n;
  // CHECK: 7
  printf("%d\n", n);
  n = m ^ n;
  // CHECK: 5
  printf("%d\n", n);
  n = m & n;
  // CHECK: 0
  printf("%d\n", n);
  return 0;
}
