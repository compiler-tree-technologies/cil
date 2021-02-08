// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

enum ENUM {
  A,
  B,
  C,
  D
};

int main() {
  enum ENUM E;
  E = C;

  if (E == A)
    printf ("A\n");
  else if (E == B)
    printf ("B\n");
  else if (E == C)
    // CHECK: C
    printf ("C\n");
  else if (E == D)
    printf ("D\n");
  else
    printf ("Unknown\n");
  return 0;
}
