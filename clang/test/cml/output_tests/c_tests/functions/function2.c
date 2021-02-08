// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int random_func(int *a) {
  a[1] = 10;
  return 0;
}

int main() {
  int a[10];
  random_func(a);
  printf("%d\n",a[1]);
  return 0;
}
//CHECK: 10
