#![cfg_attr(all(test, feature = "unstable"), feature(test))]
#![cfg_attr(all(test, feature = "unstable"), feature(iter_arith))]

#[cfg(all(test, feature = "unstable"))] extern crate test;

#[macro_use]
extern crate either;

extern crate num;
extern crate rand;

use either::{Either, Left, Right};
use num::Float;
use rand::{Rng, XorShiftRng, weak_rng};
use std::mem::replace;


/// A set of possible criteria to decide when to stop the optimization.
pub enum StopCriterion {
    /// Stop if a specified number of function evaluations has been reached.
    MaxFunctionEvaluations(usize),

    /// Stop if a specified number of iterations has been reached.
    MaxIterations(usize),

    /// Stop if a certain target value range has been reached.
    ///
    /// The first value represents the desired **target value**, while the second one
    /// specifies an **absolute error** which is additionally applied to both directions.
    TargetValue(f64, f64)
}


/// Groups information relevant to a solution.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The set of parameters supplied to the objective function.
    pub xs: Vec<f64>,

    /// Target value for the set of parameters.
    pub y: f64,

    /// Number of function evaluations.
    pub evals: usize,

    /// Number of iterations.
    pub iters: usize
}


/// Provide means to adjust the main paramters driving the algorithm.
pub struct Config {
    foods: usize,
    max_trials: usize,
    max_evals: Option<usize>,
    max_iters: Option<usize>,
    value_range: Option<(f64, f64)>
}

impl Config {
    /// Create a new `Config` which describes the behavior of the ABC algorithm.
    ///
    /// By default, this configuration provides sane defaults, i.e., it uses 20 food
    /// sources, a limit of 100 trials per food source
    pub fn new() -> Config {
        Config {
            foods: 20,
            max_trials: 100,
            max_evals: None,
            max_iters: None,
            value_range: None
        }
    }

    /// Set the number of food sources.
    pub fn foods(&mut self, foods: usize) -> &mut Config {
        assert!(foods > 0, "There must be at least 1 food source");
        self.foods = foods;
        self
    }

    /// Set the number of maximum trials per food source, bevore abondining the
    /// source.
    pub fn max_trials(&mut self, max_trials: usize) -> &mut Config {
        self.max_trials = max_trials;
        self
    }

    /// Enable the supplied stop criterion for future optimizations.
    pub fn stop(&mut self, criterion: StopCriterion) -> &mut Config {
        match criterion {
            MaxFunctionEvaluations(v) => self.max_evals = Some(v),
            MaxIterations(v) => self.max_iters = Some(v),
            TargetValue(v, e) => self.value_range = Some((v - e, v + e))
        }
        self
    }

    /// Optimize the given objective function by using the ABC algorithm with
    /// the specified parameters.
    pub fn optimize<F>(&self, dimensions: usize, low: f64, high: f64, objective: F) -> Solution
        where F: Fn(&[f64]) -> f64
    {
        assert!(dimensions > 0, "`dimensions` must be greater than zero");
        assert!(low.is_finite(), "`low` must be a finite number");
        assert!(high.is_finite(), "`high` must be a finite number");
        assert!(low < high, "`low` must be less than `high`");

        Optimizer::new(self, dimensions, low, high, &objective).run().left().unwrap()
    }
}


struct Optimizer<'a> {
    rng: XorShiftRng,
    config: &'a Config,
    dimensions: usize,
    low: f64,
    high: f64,
    objective: &'a Fn(&[f64]) -> f64,
    candidates: Vec<Candidate>,
    optimum: Option<Candidate>,
    evals: usize,
    iter: usize
}

impl<'a> Optimizer<'a> {
    fn new(config: &'a Config, dimensions: usize, low: f64, high: f64,
        objective: &'a Fn(&[f64]) -> f64) -> Optimizer<'a>
    {
        Optimizer {
            rng: weak_rng(),
            config: config,
            dimensions: dimensions,
            low: low,
            high: high,
            objective: objective,
            candidates: Vec::with_capacity(config.foods),
            optimum: None,
            evals: 0,
            iter: 0
        }
    }

    fn run(mut self) -> Either<Solution, ()> {
        self = try_right!(self.initialize());

        loop {
            self = try_right!(self.send_employed_bees());
            self = try_right!(self.send_onlooker_bees());
            self = try_right!(self.send_scout_bees());

            if self.config.max_iters.map_or(false, |max_iters| max_iters == self.iter + 1) {
                return Left(self.finish());
            }

            self.iter += 1;
        }
    }

    fn initialize(mut self) -> Either<Solution, Self> {
        for _ in 0..self.config.foods {
            self = try_right!(self.new_candidate(None));
        }

        Right(self)
    }

    fn send_employed_bees(mut self) -> Either<Solution, Self> {
        for i in 0..self.config.foods {
            self = try_right!(self.mutate_candidate(i));
        }

        // calculating probabilities
        let max_fitness = self.candidates.iter().fold(self.candidates[0].fitness,
            |acc, candidate| acc.max(candidate.fitness));

        for candidate in &mut self.candidates {
            candidate.probability = 0.9 * candidate.fitness / max_fitness + 0.1;
        }

        Right(self)
    }

    fn send_onlooker_bees(mut self) -> Either<Solution, Self> {
        let mut s = 0;
        let mut t = 0;

        while t < self.config.foods {
            if self.rng.gen::<f64>() < self.candidates[s].probability {
                t += 1;

                self = try_right!(self.mutate_candidate(s));
            }

            s += 1;
            if s == self.config.foods {
                s = 0;
            }
        }

        Right(self)
    }

    fn send_scout_bees(mut self) -> Either<Solution, Self> {
        let max_trials = self.candidates.iter().map(|c| c.trials).max().unwrap();

        if max_trials >= self.config.max_trials {
            for i in 0..self.config.foods {
                if self.candidates[i].trials == max_trials {
                    self = try_right!(self.new_candidate(Some(i)));
                }
            }
        }

        Right(self)
    }



    fn new_candidate(mut self, idx: Option<usize>) -> Either<Solution, Self> {
        let xs: Vec<_> = (0..self.dimensions).map(|_| {
            let x = self.rng.gen_range(self.low, self.high);
            assert!(x.is_finite(), "New random value is not finite!");
            x
            }).collect();
        let y = (self.objective)(&xs);
        let fitness = calc_fitness(y);

        self.evals += 1;

        let candidate = Candidate {
            xs: xs,
            y: y,
            fitness: fitness,
            probability: 0.0,
            trials: 0
        };

        let i = match idx {
            Some(i) => {
                self.candidates[i] = candidate;
                i
            },
            None => {
                self.candidates.push(candidate);
                self.candidates.len() - 1
            }
        };

        self.evaluate_candidate(i)
    }

    fn mutate_candidate(mut self, index: usize) -> Either<Solution, Self> {
        let rand_value = self.rng.gen_range(0, self.dimensions);
        let rand_candidate = (0..).map(|_| self.rng.gen_range(0, self.config.foods))
            .filter(|&other| other != index)
            .next().unwrap();

        let new_value = (self.candidates[index].xs[rand_value] +
            (self.candidates[index].xs[rand_value] - self.candidates[rand_candidate].xs[rand_value]) *
            (self.rng.gen::<f64>() - 0.5) * 2.0)
            .min(self.high).max(self.low);

        assert!(new_value.is_finite(), "Mutated value is not finite!");

        let old_value = replace(&mut self.candidates[index].xs[rand_value], new_value);

        let y = (self.objective)(&(self.candidates[index].xs));
        let fitness = calc_fitness(y);

        self.evals += 1;

        {
            let candidate = &mut self.candidates[index];
            if candidate.fitness < fitness {
                candidate.y = y;
                candidate.fitness = fitness;
                candidate.trials = 0;
            } else {
                candidate.xs[rand_value] = old_value;
                candidate.trials += 1;
            }
        }

        self.evaluate_candidate(index)
    }

    fn evaluate_candidate(mut self, idx: usize) -> Either<Solution, Self> {
        {
            let candidate = &self.candidates[idx];

            // upldate global optimum, if necessary
            if self.optimum.as_ref().map_or(true, |optimum| candidate.y < optimum.y) {
                self.optimum = Some(candidate.clone());
            }
        }

        // abort if the maximum
        if self.config.max_evals.map_or(false, |max_evals| max_evals == self.evals) ||
            self.config.value_range.map_or(false, |(l, h)| self.optimum.as_ref().map_or(false,
                |optimum| l <= optimum.y && optimum.y <= h))
        {
            return Left(self.finish());
        }

        Right(self)
    }

    fn finish(self) -> Solution {
        let optimum = self.optimum.unwrap();

        Solution {
            xs: optimum.xs,
            y: optimum.y,
            evals: self.evals,
            iters: self.iter + 1
        }
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    xs: Vec<f64>,
    y: f64,
    fitness: f64,
    probability: f64,
    trials: usize
}


#[inline(always)]
fn calc_fitness(y: f64) -> f64 {
    if y >= 0.0 {
        1.0 / (1.0 + y)
    } else {
        1.0 + y.abs()
    }
}

// re-exports
pub use StopCriterion::*;


#[cfg(all(test, feature = "unstable"))]
mod benches {
    use test::Bencher;

    use super::*;
    use super::problems::*;

    macro_rules! benches {
        ($($name: ident, $d: expr, $l: expr, $u: expr, $y: expr, $e: expr, $xs: expr, $f: ident;)*) => {
            $(
                #[bench]
                fn $name(b: &mut Bencher) {
                    let optimum = $xs;
                    assert!($d == optimum.len());

                    let mut abc = Config::new();
                    abc.stop(TargetValue($y, $e));

                    b.iter(|| {
                        let solution = abc.optimize($d, $l, $u, &$f);

                        for i in 0..$d {
                            assert!((solution.xs[i] - optimum[i]).abs() < 1.0e-5);
                        }

                        solution
                    })
                }
            )*
        }
    }

    benches! {
        sphere_01d, 1, -5.12, 5.12, 0.0, 1.0e-10, vec![0.0f64; 1], sphere;
        sphere_02d, 2, -5.12, 5.12, 0.0, 1.0e-10, vec![0.0f64; 2], sphere;
        sphere_04d, 4, -5.12, 5.12, 0.0, 1.0e-10, vec![0.0f64; 4], sphere;
        sphere_08d, 8, -5.12, 5.12, 0.0, 1.0e-10, vec![0.0f64; 8], sphere;
        sphere_16d, 16, -5.12, 5.12, 0.0, 1.0e-10, vec![0.0f64; 16], sphere;

        ackley_01d, 1, -32.768, 32.768, 0.0, 1.0e-10, vec![0.0f64; 1], ackley;
        ackley_02d, 2, -32.768, 32.768, 0.0, 1.0e-10, vec![0.0f64; 2], ackley;
        ackley_04d, 4, -32.768, 32.768, 0.0, 1.0e-10, vec![0.0f64; 4], ackley;
        ackley_08d, 8, -32.768, 32.768, 0.0, 1.0e-10, vec![0.0f64; 8], ackley;
        ackley_16d, 16, -32.768, 32.768, 0.0, 1.0e-10, vec![0.0f64; 16], ackley;

        booth_2d, 2, -10.0, 10.0, 0.0, 1.0e-10, vec![1.0, 3.0], booth;
    }
}


#[cfg(all(test, feature = "unstable"))]
mod problems {
    use std::f64::consts::PI;

    // bowl-shaped
    pub fn sphere(xs: &[f64]) -> f64 {
        xs.iter().map(|v| v * v).sum()
    }

    // many local minima
    pub fn ackley(xs: &[f64]) -> f64 {
        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * PI;

        -a * (-b * (xs.iter().map(|&x| x.powi(2)).sum::<f64>() / xs.len() as f64).sqrt()).exp() -
            (xs.iter().map(|&x| (c*x).cos()).sum::<f64>() / xs.len() as f64).exp() + a + 1.0f64.exp()
    }

    // plate-shaped
    pub fn booth(xs: &[f64]) -> f64 {
        assert!(xs.len() == 2);

        (xs[0] + 2.0*xs[1] - 7.0).powi(2) + (2.0*xs[0] + xs[1] - 5.0).powi(2)
    }
}
