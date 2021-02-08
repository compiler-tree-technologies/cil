// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a[10];
  int *b[10];
  b[3] = &a[0];
  int **c = &b[3];
  **c = 10;
  printf("%d\n",a[0]);
  return 0;
}
//CHECK: 10
