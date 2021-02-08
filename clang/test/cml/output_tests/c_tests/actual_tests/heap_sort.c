// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}
// function build Max Heap where value
// of each child is always smaller
// than value of their parent
void buildMaxHeap(int arr[], int n) {
  for (int i = 1; i < n; i++) {
    // if child is bigger than parent
    if (arr[i] > arr[(i - 1) / 2]) {
      int j = i;

      // swap child and parent until
      // parent is smaller
      while (arr[j] > arr[(j - 1) / 2]) {
        swap(&arr[j], &arr[(j - 1) / 2]);
        j = (j - 1) / 2;
      }
    }
  }
}

void heapSort(int arr[], int n) {
  buildMaxHeap(arr, n);

  for (int i = n - 1; i > 0; i--) {
    // swap value of first indexed
    // with last indexed
    swap(&arr[0], &arr[i]);

    // maintaining heap property
    // after each swapping
    int j = 0, index;

    do {
      index = (2 * j + 1);

      // if left child is smaller than
      // right child point index variable
      // to right child
      if (arr[index] < arr[index + 1] && index < (i - 1))
        index++;

      // if parent is smaller than child
      // then swapping parent with child
      // having higher value
      if (arr[j] < arr[index] && index < i)
        swap(&arr[j], &arr[index]);

      j = index;

    } while (index < i);
  }
}

// Driver Code to test above
int main() {
  int arr[6];
  arr[0] = 10;
  arr[1] = 20;
  arr[2] = 15;
  arr[3] = 17;
  arr[4] = 9;
  arr[5] = 21;
  int n = 6;

  printf("Given array: ");
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);

  printf("\n\n");

  heapSort(arr, n);

  // print array after sorting
  printf("Sorted array: ");
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);
  // CHECK: 9 10 15 17 20 21
  return 0;
}
