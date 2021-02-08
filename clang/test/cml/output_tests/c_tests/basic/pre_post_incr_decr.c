// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3;
  a++;
  printf("%d\n",a);  // CHECK: 4
  ++a;
  printf("%d\n",a);  // CHECK: 5
  a--;
  printf("%d\n",a);  // CHECK: 4
  --a;
  printf("%d\n",a);  // CHECK: 3
  return 0;
}
