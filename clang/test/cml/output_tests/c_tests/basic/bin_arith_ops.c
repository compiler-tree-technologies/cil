// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a, b;
  a = 40;
  b = 20;
  int add = a + b;
  int sub = a - b;
  int mul = a * b;
  int div = a / b;
  printf("%d %d %d %d\n",add,sub,mul,div);
  return 0;
}
// CHECK: 60 20 800 2
