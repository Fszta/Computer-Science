### Bubble sort

Bubble sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.


!!! warning

    Bubble sort is not suitable to sort large datasets

    

Implementation

=== "Go"

    ``` golang
    func bubbleSort(integers []int) []int {
        nbElements := len(integers)

        for i := 0; i < nbElements-1; i++ {
            for j := 0; j < nbElements-i-1; j++ {
                if integers[j] > integers[j+1] {
                    integers[j], integers[j+1] = integers[j+1], integers[j]
                }
            }
	    }
	    return integers
    }
    ```

=== "Python"

    ```python
    def bubble_sort():
        pass
    ```


Time complexity

$$
{O(n^2)}
$$


Space complexity
${O(1)}$
