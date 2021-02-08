// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int *p = NULL;
  int a = 10;
  p = &a;
  if (p && a)
    // CHECK: 10
    printf ("%d", a);
  else
    printf ("11");
  return 0;
}
