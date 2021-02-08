// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int vinay(int **d) {
  **d = 10;
  return 0;
}

int main() {
  int a[10];
  int *b[10];
  b[3] = &a[0];
  int **c = &b[3];
  vinay(c);
  printf("%d\n",a[0]);
  return 0;
}
//CHECK: 10
