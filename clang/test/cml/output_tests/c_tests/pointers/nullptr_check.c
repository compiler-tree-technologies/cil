// RUN: %cml %s -o %t -O0 && %t | FileCheck %s
#include <stdio.h>

int getVal(int *p) { return (*p) > 10; }

int main() {
  int *p = NULL;
  int a = 10;
  if (a > 10)
    p = &a;
  if (p && getVal(p))
    printf("Greater than 10");
  else
    // CHECK: Not greater than 10
    printf("Not greater than 10");

  return 0;
}
