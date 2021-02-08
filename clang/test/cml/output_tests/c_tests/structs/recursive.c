#include <stdio.h> 
#include <stdlib.h>
// RUN: %cml %s -o %t && %t | FileCheck %s

struct Node { 
    int data; 
    struct Node* next; 
}; 

int main() {
  struct Node head;
  head.data = 10;
  // CHECK: Head 10
  printf ("Head %d\n", head.data);
  return 0;
}
