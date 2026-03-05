use tch::{
    Kind, Tensor,
    nn::{self, ModuleT, OptimizerConfig},
};

#[derive(Debug)]
struct Model {
    vs: nn::VarStore,
    model: CNNModel,
    opt: nn::Optimizer,
}

impl Model {
    pub fn new(lr: f64) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let model = CNNModel::new(&vs.root());
        let opt = nn::Sgd::default().build(&vs, lr).unwrap();

        return Model { vs, model, opt };
    }

    pub fn train_batch(&mut self, xs: &Tensor, ys: &Tensor) -> f64 {
        let logits = self.model.forward_t(xs, true);
        let loss = logits.cross_entropy_for_logits(ys);
        self.opt.backward_step(&loss);

        loss.double_value(&[])
    }

    pub fn predict(&self, xs: &Tensor) -> Tensor {
        let logits = self.model.forward_t(xs, false);
        logits.softmax(-1, Kind::Float)
    }

    pub fn predict_classes(&self, xs: &Tensor) -> Tensor {
        let logits = self.model.forward_t(xs, false);
        logits.argmax(-1, false)
    }
}

#[derive(Debug)]
pub struct CNNModel {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    l3: nn::Linear,
    l4: nn::Linear,
}

impl CNNModel {
    pub fn new(vs: &nn::Path) -> CNNModel {
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        let conv1 = nn::conv2d(vs, 1, 32, 3, conv_cfg);
        let conv2 = nn::conv2d(vs, 32, 64, 3, conv_cfg);
        let l3 = nn::linear(vs, 3136, 128, Default::default());
        let l4 = nn::linear(vs, 128, 10, Default::default());

        CNNModel {
            conv1,
            conv2,
            l3,
            l4,
        }
    }
}

impl nn::ModuleT for CNNModel {
    fn forward_t(&self, xs: &tch::Tensor, _train: bool) -> tch::Tensor {
        xs.apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .flat_view()
            .apply(&self.l3)
            .relu()
            .apply(&self.l4)
    }
}
