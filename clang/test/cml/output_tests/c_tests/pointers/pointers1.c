// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a = 10;
  int *b;
  b = &a;
  printf("%d\n",*b);
  return 0;
}
//CHECK: 10
