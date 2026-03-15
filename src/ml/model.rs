use tch::{
    Kind, Tensor,
    nn::{self, ModuleT, OptimizerConfig},
};

use crate::config::config::CONFIG;

#[derive(Debug)]
pub struct Model {
    model: CNNModel,
    opt: nn::Optimizer,
}

impl Model {
    pub fn new(lr: f64) -> Self {
        let vs = nn::VarStore::new(CONFIG.device);
        let model = CNNModel::new(&vs.root());
        let opt = nn::Sgd::default().build(&vs, lr).unwrap();

        return Model { model, opt };
    }

    pub fn model(&self) -> Tensor {
        self.model.as_tensor()
    }

    pub fn train_batch(&mut self, xs: &Tensor, ys: &Tensor) -> f64 {
        let logits = self.model.forward_t(xs, true);
        let loss = logits.cross_entropy_for_logits(ys);
        self.opt.backward_step(&loss);

        loss.double_value(&[])
    }

    pub fn update_after_aggregation(&mut self, update: &Tensor) {
        self.model.update_from_tensor(update)
    }

    pub fn predict(&self, xs: &Tensor) -> Tensor {
        let logits = self.model.forward_t(xs, false);
        logits.softmax(-1, Kind::Float)
    }

    pub fn predict_classes(&self, xs: &Tensor) -> Tensor {
        let logits = self.model.forward_t(xs, false);
        logits.argmax(-1, false)
    }

    pub fn eval(&self, (xs, ys): (&Tensor, &Tensor)) -> f64 {
        let predictions = self.predict_classes(xs);
        let correct = predictions.eq_tensor(&ys);
        let accuracy = correct.to_kind(tch::Kind::Float).mean(tch::Kind::Float);
        accuracy.double_value(&[])
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

    pub fn as_tensor(&self) -> Tensor {
        let params: Vec<Tensor> = vec![
            self.conv1.ws.copy(),
            self.conv1.bs.as_ref().unwrap().copy(),
            self.conv2.ws.copy(),
            self.conv2.bs.as_ref().unwrap().copy(),
            self.l3.ws.copy(),
            self.l3.bs.as_ref().unwrap().copy(),
            self.l4.ws.copy(),
            self.l4.bs.as_ref().unwrap().copy(),
        ];

        Tensor::cat(
            &params.iter().map(|p| p.flatten(0, -1)).collect::<Vec<_>>(),
            0,
        )
    }

    pub fn shapes(&self) -> Vec<Vec<i64>> {
        vec![
            self.conv1.ws.size(),
            self.conv1.bs.as_ref().unwrap().size(),
            self.conv2.ws.size(),
            self.conv2.bs.as_ref().unwrap().size(),
            self.l3.ws.size(),
            self.l3.bs.as_ref().unwrap().size(),
            self.l4.ws.size(),
            self.l4.bs.as_ref().unwrap().size(),
        ]
    }

    pub fn update_from_tensor(&mut self, flat_params: &Tensor) {
        tch::no_grad(|| {
            let mut start = 0;
            let mut update_param = |t: &Tensor| -> Tensor {
                let n = t.numel() as i64;
                let slice = flat_params.narrow(0, start, n).view(&*t.size());
                start += n;
                slice
            };
            self.conv1.ws.copy_(&update_param(&self.conv1.ws));
            let conv1_bs_update = update_param(self.conv1.bs.as_ref().unwrap());
            self.conv1.bs.as_mut().unwrap().copy_(&conv1_bs_update);

            self.conv2.ws.copy_(&update_param(&self.conv2.ws));
            let conv2_bs_update = update_param(self.conv2.bs.as_ref().unwrap());
            self.conv2.bs.as_mut().unwrap().copy_(&conv2_bs_update);

            self.l3.ws.copy_(&update_param(&self.l3.ws));
            let l3_bs_update = update_param(self.l3.bs.as_ref().unwrap());
            self.l3.bs.as_mut().unwrap().copy_(&l3_bs_update);

            self.l4.ws.copy_(&update_param(&self.l4.ws));
            let l4_bs_update = update_param(self.l4.bs.as_ref().unwrap());
            self.l4.bs.as_mut().unwrap().copy_(&l4_bs_update);
        });
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
