// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a;
  a = 31;
  printf("%d\n",a);
  return 0;
}

// CHECK: 31 
