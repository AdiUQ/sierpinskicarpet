This is the sierpinski carpet generated using PyTorch.

The sierpinski carpet is a fractal that is visually different in comparison to the mandelbrot set.
The differences occur as a square is repeatedly divided into a 3x3 grid whilst removing the centre square.
This process is repeated on the remaining 8 squares. 
As the number of iterations increases indefinitely, the self similar structure of the carpet appears. 


# how my implementation works

- carpet sizing is calculated using 3 ** iterations as each iteration divides the coordinate space into thirds.

- pytorch tensors create xy coords of every point in hte img
- torch.meshgrid() combines xy coord arrays 2 produce 2dtensor coords

- the carpet initially has every pixel especially in carpet = torch.ones(..)
- t each iteration, the following condition identifies pixels that belong to the centre square:

`(x_temp % 3 == 1) & (y_temp % 3 == 1)`

These pixels are removed from the carpet using:

`carpet &= ~center_square`

The coordinates are then divided by 3:

`x_temp //= 3`

`y_temp //= 3`

This allows the same center-square removal to be performed at the next larger coord repr.

Repeating this creates a sierpinski carpet. 



## Why PyTorch Is a Major Part of the Algorithm

PyTorch is used for the coordinate grids, Boolean masks and the operations that determine which pixels remain in the Sierpinski carpet.

Instead of using nested Python loops to individually test every pixel, the algorithm performs operations on entire PyTorch tensors.

For example:

`center_square = (x_temp % 3 == 1) & (y_temp % 3 == 1)`

performs the centre-square test across the coordinate tensors at once.

Therefore, PyTorch is being used as part of the actual fractal-generation algorithm rather than only being used to display the result.

---

## Parallelism

The fractal is suitable for parallel computation because the centre-square test for one pixel does not depend on the result for another pixel.

PyTorch can therefore apply the same tensor operations across many coordinates in parallel.

When a CUDA-compatible GPU is available, the tensors are transferred to the GPU and these operations can be executed using GPU parallelism.

The device is selected using:

`torch.device("cuda" if torch.cuda.is_available() else "cpu")`

This means the implementation can also fall back to the CPU when CUDA is unavailable.

---

## Fractal Dimension

The theoretical fractal dimension of the Sierpinski carpet is:

`D = log(8) / log(3)`

which gives approximately:

`D = 1.8928`

The value 8 comes from the 8 remaining self-similar squares after the centre square is removed.

The value 3 comes from each copy having a scale of one third of the original square.

The dimension being between 1 and 2 reflects that the Sierpinski carpet is more complex than a one-dimensional line but does not completely fill a two-dimensional plane.

---

## Iteration Depth Analysis

I generated the Sierpinski carpet at iteration depths 2, 4 and 6.

At low iteration depths, only the larger removed squares are visible.

As the iteration depth increases, the same center removal pattern occurs at increasingly smaller scales.

This demonstrates the selfsimilarityfeatures

The size of the generated coordinate grid also grows as:

`3 ** iterations`

For example:

- Iteration 2: 9 × 9
- Iteration 4: 81 × 81
- Iteration 6: 729 × 729

Therefore, increasing the iteration depth significantly increases the number of pixels processed by the tensor operations.

---

## Alternative Visualisation

In addition to the standard grayscale representation, I generated an alternative visualisation using Matplotlib's `viridis` colour map.

The underlying Sierpinski carpet is unchanged. Only the way the Boolean tensor values are mapped to colours is different.

This demonstrates that the fractal structure is produced by the PyTorch algorithm independently of how the result is visualised.

---

## AI Usage

AI assistance was used during the development of this project.

The prompts and relevant outputs/reasoning used during development have been documented as required by the task specification.

I used the generated material as a starting point and worked through the implementation to understand the tensor operations, fractal-generation rule, PyTorch parallelism, iteration-depth behaviour and theoretical fractal dimension.
