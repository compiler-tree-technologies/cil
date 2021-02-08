// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a[10][20];
  a[9][19] = 3;
  a[0][0] = 11;
  printf("%d %d\n", a[9][19], a[0][0]);
  return 0;
}
//CHECK: 3 11
