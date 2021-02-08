// A simple C program to introduce
// a linked list
// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

struct Node {
  int data;
  struct Node *next;
};

int main() {
  struct Node *head;
  struct Node *second;
  struct Node *third;

  head = malloc(sizeof(struct Node));
  second = malloc(sizeof(struct Node));
  third = malloc(sizeof(struct Node));

  head->data = 1;      // assign data in first node
  head->next = second; // Link first node with
  second->data = 2;

  second->next = third;
  third->data = 3; // assign data to third node
  third->next = NULL;

  // CHECK: 1->2->3
  printf("%d->%d->%d\n", head->data, second->data, third->data);

  free(head);
  free(second);
  free(third);

  return 0;
}
