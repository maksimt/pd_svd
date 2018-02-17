# SVD using block power iterations

### Usage
Note: this is not going to work until we include local dependencies.

```bash
git clone https://github.com/maksimt/pd_svd
luigi --module pd_svd.learning_outcome_expm EvalAll --workers 2 --local-scheduler
```

### Simulating fixed precision arithmetic

The purpose of this library is to eventually rely on
using cryptographic primitives for multiparty computation.
For now we simulate loss of numeric precision.

The crypto primitives we use work in a finite field.
As a result we have to encode our floats to integers, and then decode them back.
To encode a float `f` as an int `x` we say `x=f*(1<<n_bits)`.
To decode a int `x` into a float `f` we say `f=x/(1<<n_bits)`.
However in order to achieve reasonable precision we need `n_bits` around 20.
With large input matrices this is sometimes not possible
without causing overflow errors when working with `64bit` integers.
This is because the largest value we can represent is `2^63-1`.
The bottleneck comes from [finding the 2-norm](https://github.com/maksimt/pd_svd/blob/master/blockpower_svd.py#L76) of the columns of `V`,
which we do by squaring entries and adding them up.

To overcome this issue we simulate `n_bits` of precision by [rounding](https://github.com/maksimt/pd_svd/blob/master/blockpower_svd.py#L17)
floats to appropriately many decimal places.
 
![eq](https://latex.codecogs.com/gif.latex?\LARGE&space;\begin{align*}&space;n_{bits}&=\lceil\frac{\log(10^d)}{\log&space;2}\rceil\\&space;d&\approx\frac{\log(e^{\log(2)n_{bits}})}{\log&space;10}&space;\end{align*})

In actual crypto we would potententially work with 128 or 256 bit integers,
or think about how to reduce the magnitude of the floats in the matrices we are operating on.

Even with 20-bits of precision we are able to reconstruct a right singular vector 
with less than `1e-5` reconstruction error.
Downstream applications like principal component regression and low rank approximation
don't seem to be affected as compared to `sklearn`'s `randomized_svd()`.
