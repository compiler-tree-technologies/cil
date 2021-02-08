// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

enum ENUM {
  A = 11,
  B = 12,
  C = 13,
  D = 14
};

int main() {
  enum ENUM E;
  E = A;

  if (E == A)
    // CHECK: A
    printf ("A\n");
  else if (E == B)
    printf ("B\n");
  else if (E == C)
    printf ("C\n");
  else
    printf ("Unknown\n");
  return 0;
}
