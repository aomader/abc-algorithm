# abc-algorithm [![Build Status](https://travis-ci.org/b52/abc-algorithm.svg?branch=master)](https://travis-ci.org/b52/abc-algorithm) [![Coverage Status](https://coveralls.io/repos/b52/abc-algorithm/badge.svg?branch=master&service=github)](https://coveralls.io/github/b52/abc-algorithm?branch=master)

A [Rust] implementation of the [Artificial Bee Colony (ABC) Algorithm] by
Karaboga [[1]].

## Usage

```Rust
#![feature(iter_arith)]

extern crate abc_algorithm;

use std::f64::consts::PI;

fn main() {
    let solution = abc_algorithm::Config::new()
        .stop(abc_algorithm::StopCriterion::TargetValue(0.0, 1.0e-10))
        .optimize(2, -5.0, 5.0, &ackley);

    println!("Best solution: {:?}", solution);
}

fn ackley(xs: &[f64]) -> f64 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    -a * (-b * (xs.iter().map(|&x| x.powi(2)).sum::<f64>() / xs.len() as f64).sqrt()).exp() -
        (xs.iter().map(|&x| (c*x).cos()).sum::<f64>() / xs.len() as f64).exp() + a + 1.0f64.exp()
}
```

## License

This software is licensed under the terms of the MIT license. Please see the
[LICENSE](LICENSE) for full details.

[Rust]: http://www.rust-lang.org/
[Artificial Bee Colony (ABC) Algorithm]: http://mf.erciyes.edu.tr/abc/
[[1]]: http://www-lia.deis.unibo.it/Courses/SistInt/articoli/bee-colony1.pdf
