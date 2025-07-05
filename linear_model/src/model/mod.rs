pub struct Dense {
    pub weight: Vec<f32>,
    pub bias: f32,
}

impl Dense {
    pub fn new(in_features: i32, out_features: i32) -> Self {
        let upper = 1.0 / (in_features as f64).sqrt();
        let lower = -upper;

        Self {
            weight: vec![1.0, 2.0, 3.0],
            bias: 0.1,
        }
    }
}
