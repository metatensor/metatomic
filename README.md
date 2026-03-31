<h1>
<p align="center">
    <img src="https://raw.githubusercontent.com/metatensor/metatomic/refs/heads/main/docs/static/images/metatomic-horizontal-dark.png" alt="Metatomic logo" width="600"/>
</p>
</h1>

<h4 align="center">

[![tests status](https://img.shields.io/github/checks-status/metatensor/metatomic/main)](https://github.com/metatensor/metatomic/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://docs.metatensor.org/metatomic/)
[![coverage](https://codecov.io/gh/metatensor/metatomic/branch/main/graph/badge.svg)](https://codecov.io/gh/metatensor/metatomic)
</h4>


``metatomic`` is a library that defines a common interface between atomistic
machine learning models, and atomistic simulation engines. Our main goal is to
define and train models once, and then be able to re-use them across many
different simulation engines (such as LAMMPS, GROMACS, *etc.*). We strive to
achieve this goal without imposing any structure on the model itself, and to
allow any model architecture to be used.


# Documentation

For details, tutorials, and examples, please have a look at our
[documentation](https://docs.metatensor.org/metatomic/).


# Contributors

Thanks goes to all people that make metatomic possible:

[![contributors list](https://contrib.rocks/image?repo=metatensor/metatomic)](https://github.com/metatensor/metatomic/graphs/contributors)

We always welcome new contributors. If you want to help us take a look at our
[contribution guidelines](CONTRIBUTING.rst) and afterwards you may start with an
open issue marked as [good first
issue](https://github.com/metatensor/metatomic/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

This project is [maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by @Luthaf and @HaoZeke, who will reply to issues and pull requests opened on this repository as soon as possible. You can mention them directly if you did not receive an answer after a couple of days.

<!-- marker-cite -->

# Citing metatomic

If you found metatomic useful for your work, please cite the corresponding article:

F. Bigi, J.W. Abbott, P. Loche et. al.<br>
Metatensor and metatomic: foundational libraries for interoperable atomistic machine learning, (2026).<br>
[https://doi.org/10.1063/5.0304911](https://doi.org/10.1063/5.0304911)

```bibtex
@article{bigi_metatensor_2026,
  title = {Metatensor and Metatomic: {{Foundational}} Libraries for Interoperable Atomistic Machine Learning},
  shorttitle = {Metatensor and Metatomic},
  author = {Bigi, Filippo and Abbott, Joseph W. and Loche, Philip and Mazitov, Arslan and Tisi, Davide and Langer, Marcel F. and Goscinski, Alexander and Pegolo, Paolo and Chong, Sanggyu and Goswami, Rohit and Febrer, Pol and Chorna, Sofiia and Kellner, Matthias and Ceriotti, Michele and Fraux, Guillaume},
  year = 2026,
  month = feb,
  journal = {J. Chem. Phys.},
  volume = {164},
  number = {6},
  pages = {064113},
  issn = {0021-9606},
  doi = {10.1063/5.0304911},
}
```

<!-- marker-end -->
