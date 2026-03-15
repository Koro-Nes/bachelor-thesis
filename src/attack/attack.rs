use std::{fmt::Display, ops::Mul};

use tch::Tensor;

use crate::{config::config::CONFIG, ml::dataset::LocalDataSet};

#[derive(Debug, Clone, Copy)]
pub enum AttackType {
    LabelFlipping,
    SignFlipping,
    BackdoorTrigger,
    NoAttack,
}

impl Display for AttackType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttackType::BackdoorTrigger => f.write_str("Backdoor"),
            AttackType::LabelFlipping => f.write_str("LabelFlip"),
            AttackType::SignFlipping => f.write_str("SignFlip"),
            AttackType::NoAttack => f.write_str("noAttack"),
        }
    }
}

pub trait AttackStrategy {
    fn manipulate_model(&self, model: &Tensor) -> Tensor;

    fn manipulate_dataset(&self, dataset: &mut LocalDataSet);

    fn kind(&self) -> AttackType;
}

#[derive(Debug)]
pub struct LabelFlippingAttack {}

impl LabelFlippingAttack {
    pub fn new() -> Self {
        LabelFlippingAttack {}
    }
}

impl AttackStrategy for LabelFlippingAttack {
    fn manipulate_model(&self, _model: &Tensor) -> Tensor {
        panic!("label flipping attack should never call manipulate_model")
    }

    fn manipulate_dataset(&self, dataset: &mut LocalDataSet) {
        for i in 0..CONFIG.labelflipping.from.len() {
            let from = CONFIG.labelflipping.from[i];
            let to = CONFIG.labelflipping.to[i];

            let mask = dataset.train_y.eq(from as i64);
            let idxs = mask.nonzero();
            let n_flip =
                ((idxs.size()[0] as f64) * CONFIG.labelflipping.flip_fraction).round() as i64;

            if n_flip > 0 {
                let perm = Tensor::randperm(idxs.size()[0], (tch::Kind::Int64, dataset.device));
                let flip_idxs = idxs
                    .index_select(0, &perm.narrow(0, 0, n_flip))
                    .squeeze_dim(1);
                dataset.train_y = dataset.train_y.index_fill(0, &flip_idxs, to as i64);
            }
        }
    }

    fn kind(&self) -> AttackType {
        AttackType::LabelFlipping
    }
}

#[derive(Debug)]
pub struct SignFlipAttack {}

impl SignFlipAttack {
    pub fn new() -> Self {
        Self {}
    }
}

impl AttackStrategy for SignFlipAttack {
    fn manipulate_model(&self, model: &Tensor) -> Tensor {
        model.mul(-1)
    }

    fn manipulate_dataset(&self, _dataset: &mut LocalDataSet) {
        panic!("SignFlipAttack should never call manipulate_dataset()")
    }

    fn kind(&self) -> AttackType {
        AttackType::SignFlipping
    }
}

#[derive(Debug)]
pub struct BackDoorTriggerAttack {}

impl BackDoorTriggerAttack {
    pub fn new() -> Self {
        Self {}
    }

    pub fn poison_test_set(&self, test_xs: &Tensor, test_ys: &Tensor) -> (Tensor, Tensor) {
        let test_xs = test_xs.copy();
        let test_ys = test_ys.copy();

        let (_, _, h, w) = test_xs.size4().unwrap();
        let val = CONFIG.backdoor.trigger_value;
        let target = CONFIG.backdoor.target_label;
        let size = CONFIG.backdoor.trigger_size;

        tch::no_grad(|| {
            for i in 0..test_xs.size()[0] {
                let img = test_xs.get(i);
                let _ = img
                    .narrow(1, h - size - 1, size)
                    .narrow(2, w - size - 1, size)
                    .fill_(val);
                let _ = test_ys.get(i).fill_(target);
            }
        });
        (test_xs, test_ys)
    }
}

impl AttackStrategy for BackDoorTriggerAttack {
    fn manipulate_model(&self, _model: &Tensor) -> Tensor {
        panic!("BackdoorTriggerAttack should never call manipulate_model()")
    }

    fn manipulate_dataset(&self, dataset: &mut LocalDataSet) {
        let num_train = dataset.train_x.size()[0];
        let num_to_poison = (num_train as f64 * CONFIG.backdoor.poison_fraction) as i64;

        let x = &dataset.train_x;
        let y = &dataset.train_y;

        let (_, _, h, w) = x.size4().unwrap();

        let val = CONFIG.backdoor.trigger_value;
        let target = CONFIG.backdoor.target_label;
        let size = CONFIG.backdoor.trigger_size;

        tch::no_grad(|| {
            for i in 0..num_to_poison {
                let img = x.get(i);

                let _ = img
                    .narrow(1, h - size - 1, size)
                    .narrow(2, w - size - 1, size)
                    .fill_(val);

                let _ = y.get(i).fill_(target);
            }
        })
    }

    fn kind(&self) -> AttackType {
        AttackType::BackdoorTrigger
    }
}
