// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a[2];
  a[1] = 11;
  int *b =  &a[0];
  b = b + 1;
  *b = 2;
  printf("%d\n",a[1]);
  return 0;
}
//CHECK: 2
