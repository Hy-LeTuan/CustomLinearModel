use crate::model::Compute;
use crate::model::dense::Dense;
use std::fmt;

pub struct Model {
    layers: Vec<Dense>,
}

impl Model {
    pub fn new(dims: Vec<usize>, output: usize) -> Self {
        let mut layers = Vec::new();

        for i in 1..dims.len() {
            let dense = Dense::new(dims[i - 1], dims[i], format!("Dense {}", i));
            layers.push(dense);
        }

        layers.push(Dense::new(
            dims[dims.len() - 1],
            output,
            String::from("Output layer"),
        ));

        return Self { layers: layers };
    }

    pub fn get_weight(
        &self,
        i: usize,
    ) -> Option<&ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>> {
        // no negative indexing in Rust
        if i >= self.layers.len() {
            return None;
        } else {
            return Some(self.layers[i].get_weight());
        }
    }
}

impl Compute for Model {
    fn compute_single(
        &self,
        x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2> {
        let mut result = x;

        for i in 0..self.layers.len() {
            result = self.layers[i].compute_single(result);
        }

        return result;
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model has {} layers\n", self.layers.len())?;
        write!(f, "-----------------\n")?;
        for i in 0..self.layers.len() {
            write!(f, "Layer {}, {}\n", i, self.layers[i])?;
            write!(f, "-----------------\n")?;
        }
        Ok(())
    }
}
