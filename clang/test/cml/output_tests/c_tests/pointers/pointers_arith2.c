// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a[10];
  int *b = &a[0];
  b[1] = 10;
  printf("%d\n",a[1]);
  return 0;
}
//CHECK: 10
