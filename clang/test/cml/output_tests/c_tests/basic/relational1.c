// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a, b;
  a = 40;
  b = 20;
  int add = 0;
  add = (a < b);
  printf("%d\n",add);
  return 0;
}
// CHECK: 0
