// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3;
  if (a >=3 )
    a++;

  for (int i = 0; i < 10; i++) {
    a = a + 3;
  }
  printf ("%d\n",a); // CHECK: 34
  return 0;
}
