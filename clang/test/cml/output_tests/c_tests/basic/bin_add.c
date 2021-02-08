// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a, b;
  a = 31;
  b = 20;
  a = a + b;
  printf("%d\n",a);
  return 0;
}
// CHECK: 51 
