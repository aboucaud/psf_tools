# Create homogenization kernel

The goal of this script is given two PSFs $P_1$ and $P_2$ with ${\rm fwhm}({P_1}) < {\rm fwhm}({P_2})$, compute the kernel $K$ such that

$$P_2 = K \ast P_1$$

If you are interested in this, please have a look at the **`pypher`** program available [here][pypher] and references therein.

[pypher]: https://github.com/aboucaud/pypher

---

::: psftools.scripts.make_kernel
