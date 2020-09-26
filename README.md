# CIAOAlgorithms.jl

[![Build status](https://github.com/kul-forbes/CIAOAlgorithms/workflows/CI/badge.svg)](https://github.com/kul-forbes/CIAOAlgorithms/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/kul-forbes/CIAOAlgorithms/coverage.svg?branch=master)](http://codecov.io/github/kul-forbes/CIAOAlgorithms?branch=master)

CIAOAlgorithms implements Block-Coordinate and Incremental Aggregated Optimization Algorithms for minimizations of the form
```math
minimize    1/N sum_{i=1}^N f_i(x) + g(x)
``` 
or 
```math
minimize    1/N sum_{i=1}^N f_i(x_i) + g(sum_{i=1}^N x_i)
``` 
where f_i are smooth, and g is (possibly) nonsmooth with easy to compute proximal mapping. These functions can be defined using the [ProximalOperators.jl](https://github.com/kul-forbes/ProximalOperators.jl) package. 

### Quick guide
You can add CIAOAlgorithms by pressing `]` to enter the package manager, then
```
pkg> add CIAOAlgorithms
```

Simple Lasso and logisitc regression test examples can be found [here](test). 

### Implemented Algorithms

Algorithm                             | Function      | Reference
--------------------------------------|---------------|-----------
Finito/MISO/DIAG  | [`Finito`](src/algorithms/Finito/Finito.jl) | [[1]][Defazio2014Finito], [[4]][Mairal2015Incremental], [[8]][Mokhtari2018Surpassing], [[9]][Latafat2019Block]
ProShI  | [`Proshi`](src/algorithms/ProShI/ProShI.jl) | [[9]][Latafat2019Block]
SAGA  | [`SAGA`](src/algorithms/SAGA_SAG/SAGA.jl) | [[3]][Defazio2014SAGA], [[6]][Defazio2014SAGA]
SAG  | [`SAG`](src/algorithms/SAGA_SAG/SAGA.jl) | [[7]][Schmidt2017Minimizing]
SVRG/SVRG++  | [`SVRG`](src/algorithms/SVRG/SVRG.jl) | [[2]][Xiao2014Proximal], [[4]][AllenZhu2016Improved], [[5]][Reddi2016Proximal]

### References

[[1]][Defazio2014Finito] Defazio, Domke, *Finito: A faster, permutable incremental gradient method for big data problems*, In International Conference on Machine Learning pp. 1125-1133 (2014).

[[2]][Xiao2014Proximal] Xiao, Zhang, *A proximal stochastic gradient method  with progressive variance reduction*, SIAM Journal on Optimization 24(4):2057–2075 (2014).

[[3]][Defazio2014SAGA] Defazio, Bach, Lacoste-Julien, *SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives*, In: Advances in neural information processing systems, pp. 1646–1654 (2014).

[[4]][Mairal2015Incremental] Mairal, *Incremental majorization-minimization optimization with application to large-scale machine learning*
SIAM Journal on Optimization 25(2), 829–855 (2015).

[[5]][AllenZhu2016Improved] Allen-Zhu, Yuan, *Improved SVRG for non-strongly-convex or sum-of-non-convex objectives* In Proceedings of the 33rd International Conference on Machine Learning 1080–1089 (2016). 

[[6]][Reddi2016Proximal] Reddi, Sra, Poczos, and Smola, *Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization* In Advances in Neural Information Processing Systems, pp. 1145–1153 (2016).

[[7]][Schmidt2017Minimizing] Schmidt, Le Roux, Bach, *Minimizing finite sums with the stochastic average gradient* Mathematical Programming, 162(1-2), 83-112 (2017).

[[8]][Mokhtari2018Surpassing] Mokhtari, Gürbüzbalaban, Ribeiro, *Surpassing gradient descent provably: A cyclic incremental method with linear convergence rate* SIAM Journal on Optimization 28(2), 1420–1447 (2018).

[[9]][Latafat2019Block] Latafat, Themelis, Patrinos, *Block-coordinate and incremental aggregated proximal gradient methods for nonsmooth nonconvex problems* arXiv:1906.10053 (2019).



[Defazio2014Finito]: https://arxiv.org/pdf/1407.2710.pdf
[Xiao2014Proximal]: https://epubs.siam.org/doi/pdf/10.1137/140961791
[Defazio2014SAGA]: https://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf
[Mairal2015Incremental]: https://epubs.siam.org/doi/pdf/10.1137/140957639
[AllenZhu2016Improved]: https://arxiv.org/pdf/1506.01972.pdf
[Reddi2016Proximal]: https://papers.nips.cc/paper/6116-proximal-stochastic-methods-for-nonsmooth-nonconvex-finite-sum-optimization.pdf
[Schmidt2017Minimizing]: https://link.springer.com/article/10.1007/s10107-016-1030-6
[Mokhtari2018Surpassing]: https://epubs.siam.org/doi/pdf/10.1137/16M1101702
[Latafat2019Block]: https://arxiv.org/pdf/1906.10053.pdf
