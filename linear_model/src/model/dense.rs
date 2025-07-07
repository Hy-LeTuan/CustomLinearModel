use crate::model::Compute;
use ndarray::prelude::*;
use rand::Rng;
use std::fmt;

#[derive(Debug)]
pub(crate) struct Dense {
    pub weights: ndarray::Array<f64, Dim<[usize; 2]>>,
    pub bias: f32,
    pub name: String,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize, name: String) -> Self {
        let upper = 6.0 / (in_features as f64 + out_features as f64).sqrt();
        let lower = -1 as f64 * upper;
        let difference = upper - lower;

        let mut weights = Array::from_elem((in_features, out_features).f(), lower as f64);

        for i in 0..in_features {
            for j in 0..out_features {
                weights[[i, j]] += rand::thread_rng().gen_range(0.0..1.0) as f64 * difference;
            }
        }

        return Self {
            weights: weights,
            bias: rand::thread_rng().gen_range(0.0..1.0),
            name: name,
        };
    }

    pub fn get_weight(&self) -> &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        return &self.weights;
    }
}

impl Compute for Dense {
    fn compute_single(
        &self,
        x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix2>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        assert_eq!(
            self.weights.shape()[0],
            x.shape()[x.shape().len() - 1],
            "Shape mismatched. Input has shape: {:?} but weight has shape: {:?}",
            x.shape(),
            self.weights.shape()
        );

        let result = x.dot(&self.weights);
        return result;
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Name: {}\nWeights: {:?}\nBias: {}\n",
            self.name, self.weights, self.bias
        )?;
        Ok(())
    }
}
