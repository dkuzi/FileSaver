```@meta
EditURL = "examples/docs_0_basics.jl"
```

#AVI.jl

This package is a toolbox for polynomial feature extraction and transformation using the Oracle Approximate Vanishing Ideal Algorithm.

##Overview

The Oracle Approximate Vanishing Ideal (OAVI) algorithm was designed to compute the vanishing ideal of a set of points. Instead of adopting the then common approach of
using singular value decomposition, OAVI finds vanishing polynomials by solving a convex optimization problem of the form
```math
\min_{\|x\|_1 \le \tau} \|Ax + b\|_2^2
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

