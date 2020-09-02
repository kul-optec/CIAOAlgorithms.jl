# CIAOAlgorithms
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
pkg> add https://github.com/kul-forbes/CIAOAlgorithms
```

Simple Lasso and logisitc regression test examples can be found [here](test). 

### Implemented Algorithms

Algorithm                             | Function      | Reference
--------------------------------------|---------------|-----------
Finito/MISO/DIAG  | [`Finito`](src/algorithms/Finito) | [[1]][Defazio2014Finito], [[3]][Mairal2015Incremental], [[6]][Mokhtari2018Surpassing], [[7]][Latafat2019Block]
ProShI  | [`Proshi`](src/algorithms/ProShI) | [[7]][Latafat2019Block]
SVRG/SVRG++  | [`SVRG`](src/algorithms/SVRG) | [[2]][Xiao2014Proximal], [[4]][AllenZhu2016Improved], [[5]][Reddi2016Proximal]

### References

[[1]][Defazio2014Finito] Defazio, Domke, *Finito: A faster, permutable incremental gradient method for big data problems*, In International Conference on Machine Learning pp. 1125-1133 (2014).

[[2]][Xiao2014Proximal] Xiao, Zhang, *A proximal stochastic gradient method  with progressive variance reduction*, SIAM Journal on Optimization 24(4):2057–2075 (2014).

[[3]][Mairal2015Incremental] Mairal, *Incremental majorization-minimization optimization with application to large-scale machine learning*
SIAM Journal on Optimization 25(2), 829–855 (2015).

[[4]][AllenZhu2016Improved] Allen-Zhu, Yuan, *Improved SVRG for non-strongly-convex or sum-of-non-convex objectives* In Proceedings of the 33rd International Conference on Machine Learning 1080–1089 (2016). 

[[5]][Reddi2016Proximal] Reddi, Sra, Poczos, and Smola, *Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization* In Advances in Neural Information Processing Systems, pp. 1145–1153 (2016).

[[6]][Mokhtari2018Surpassing] Mokhtari, Gürbüzbalaban, Ribeiro, *Surpassing gradient descent provably: A cyclic incremental method with linear convergence rate* SIAM Journal on Optimization 28(2), 1420–1447 (2018).

[[7]][Latafat2019Block] Latafat, Themelis, Patrinos, *Block-coordinate and incremental aggregated proximal gradient methods for nonsmooth nonconvex problems* arXiv:1906.10053 (2019).




[Defazio2014Finito]: https://arxiv.org/pdf/1407.2710.pdf
[Xiao2014Proximal]: https://epubs.siam.org/doi/pdf/10.1137/140961791
[Mairal2015Incremental]: https://epubs.siam.org/doi/pdf/10.1137/140957639
[AllenZhu2016Improved]: https://arxiv.org/pdf/1506.01972.pdf
[Reddi2016Proximal]: https://papers.nips.cc/paper/6116-proximal-stochastic-methods-for-nonsmooth-nonconvex-finite-sum-optimization.pdf
[Mokhtari2018Surpassing]: https://epubs.siam.org/doi/pdf/10.1137/16M1101702
[Latafat2019Block]: https://arxiv.org/pdf/1906.10053.pdf
