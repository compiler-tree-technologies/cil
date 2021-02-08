// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

struct A {
  int val;
};

int main() {
  A o;
  o.val = 10;
  printf("%d\n",o.val); // CHECK: 10
  return 0;
}
