from src.transformer.transformer import TSTransformerEncoderClassiregressor
from src.trainer.trainer import Trainer
from src.utils import get_data, get_neural_data
from src.library import *


def run_all():
    dh = get_neural_data() #(train_path='data/MotionSenseHAR/MotionSenseHAR_TRAIN.ts',
    #               test_path='data/MotionSenseHAR/MotionSenseHAR_TEST.ts')
    # print(dh.shape)
    print(dh)
    for lr in [1e-4]:
        model = TSTransformerEncoderClassiregressor(
            feat_dim=192,
            d_model=192,
            max_len=201,
            n_heads=2,
            num_layers=2,
            dim_feedforward=256,
            num_classes=31,
            dropout=0,
            pos_encoding="learnable",
            activation="gelu",
            norm="BatchNorm",
            freeze=False,
        )
        trainer = Trainer(dh=dh, epochs=8)
        dh.create_dataset()
        dh.split_data(train_split=0.827)
        dataloader_train = dh.create_dataloader(dh.train_data, batch_size=16)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.fit(dataloader=dataloader_train, model=model, optimiser=optimiser)
        dataloader_test = dh.create_dataloader(dh.test_data, batch_size=16)
        accuracy = trainer.evaluate(dataloader=dataloader_test, model=model)
        print(accuracy)


if __name__ == "__main__":
    run_all()
