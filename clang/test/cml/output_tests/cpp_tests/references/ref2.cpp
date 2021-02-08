// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

int &func(int &val, int val2) {
  val++;
  return val;
}

int main() {
  int a = 10;
  int &ref = func(a,11);
  printf("%d\n",a); // CHECK: 11
  ref++;
  printf("%d\n",a); // CHECK: 12
  return 0;
}
