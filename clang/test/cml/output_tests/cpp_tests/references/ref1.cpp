// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

void func(int &val, int val2) {
  val++;
  val2++;
}

int main() {
  int a = 10;
  func(a, a);
  printf("%d\n",a); // CHECK: 11
  return 0;
}
