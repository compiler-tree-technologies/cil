// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>


int& func(int &val) {
  return val;
}
int main() {
  int a = 10;
  int &ref = func(a);
  int *ptr = &ref;
  *ptr = 11;
  printf("%d\n",a); // CHECK: 11
  return 0;
}
