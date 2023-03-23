### Selection sort
Selection sort is an in-place comparison sorting algorithm that divides the input list into two parts: the sublist of items already sorted, and the sublist of items remaining to be sorted. It finds the minimum element in the unsorted sublist and swaps it with the leftmost unsorted element, moving the sublist boundaries one element to the right.

!!! warning

    Selection sort is not suitable to sort large datasets

### Implementation

=== "Go"

    ``` golang
    func selectionSort(integers []int) []int {
        nbElements := len(integers)

        for i := 0; i < nbElements-1; i++ {
            minIndex := i
            for j := i; j < nbElements; j++ {
                if integers[j] < integers[minIndex] {
                    minIndex = j
                }
            }
            integers[i], integers[minIndex] = integers[minIndex], integers[i]
        }

        return integers
    }
    ```

=== "Python"

    ```python
    def selection_sort():
        pass
    ```


### Time complexity

### Space complexity

!!! note

    Selection sort is useful when memory usage is a concern, as it is an in-place sorting algorithm that does not require additional memory. It is also useful when you only need to sort a small number of items, and performance is not a critical concern. Selection sort can be used when the list is partially sorted or when the order of equal elements is not important.



