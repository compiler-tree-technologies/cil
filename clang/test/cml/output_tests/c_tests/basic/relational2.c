// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a, b;
  a = 40;
  b = 20;
  int res = (a < b);
  printf("%d\n",res); // CHECK: 0
  res = (a > b);
  printf("%d\n",res); // CHECK : 1
  res = (a == b);
  printf("%d\n",res); // CHECK: 0
  res = (a != b);
  printf("%d\n",res); // CHECK : 1
  res = (a <= b);
  printf("%d\n",res); // CHECK: 0
  res = (a >= b);
  printf("%d\n",res); // CHECK : 1
  return 0;
}
