use ndarray::{ArrayBase, Ix2};

#[derive(Debug)]
pub struct Tensor {
    pub grad: f64,
    pub name: String,

    pub raw_array: ArrayBase<ndarray::OwnedRepr<f64>, Ix2>,
    pub originate: (String, String),
}

impl Tensor {
    pub fn new(
        tensor: ArrayBase<ndarray::OwnedRepr<f64>, Ix2>,
        name: String,
        originate: (String, String),
        grad: f64,
    ) -> Self {
        let new_tensor = Tensor {
            grad: grad,
            name: String::from(name),
            raw_array: tensor,
            originate: originate,
        };

        return new_tensor;
    }

    pub fn set_grad(&mut self, new_grad: f64) {
        self.grad = new_grad;
    }

    pub fn shape(&self) -> &[usize] {
        return self.raw_array.shape();
    }

    pub fn dot(&self, rhs: &Tensor) -> Tensor {
        let originate = (String::from(&self.name), String::from(&rhs.name));
        let name = format!("{}+{}", &self.name, &rhs.name);
        let tensor = self.raw_array.dot(&rhs.raw_array);

        let new_tensor = Tensor {
            grad: 0.0,
            name: name,
            raw_array: tensor,
            originate: originate,
        };

        return new_tensor;
    }
}
