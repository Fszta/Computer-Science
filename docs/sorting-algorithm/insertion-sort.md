### Insertion sort
Insertion sort is a simple sorting algorithm that builds the final sorted array one item at a time. It iterates through an input list, compares adjacent elements and inserts the current element into the correct position in the sorted list.

!!! warning

    Insertion sort is not suitable to sort large datasets


Implementation

=== "Go"

    ``` golang
    func insertionSort(integers []int) []int {
        nbElements := len(integers)

        for i := 0; i < nbElements; i++ {
            valueToInsert := integers[i]
            currentPosition := i

            for currentPosition > 0 && integers[currentPosition-1] > valueToInsert {
                integers[currentPosition] = integers[currentPosition-1]
                currentPosition = currentPosition - 1
            }
            integers[currentPosition] = valueToInsert
        }
        return integers
    }

    ```

=== "Python"

    ```python
    def insertion_sort():
        pass
    ```


Time complexity

Space complexity

