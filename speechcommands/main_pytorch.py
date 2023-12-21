import argparse
import time

import kwt_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlx.data.datasets import load_speechcommands
from mlx.data.features import mfsc

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--arch",
    type=str,
    default="kwt1",
    choices=[f"kwt{d}" for d in [1, 2, 3]],
    help="model architecture",
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def prepare_dataset(batch_size, split, root=None):
    def normalize(x):
        return (x - x.mean()) / x.std()

    data = load_speechcommands(split=split, root=root)

    data_iter = (
        data.squeeze("audio")
        .key_transform(
            "audio",
            mfsc(
                40,
                16000,
                frame_size_ms=30,
                frame_stride_ms=10,
                high_freq=7600,
                low_freq=20,
            ),
        )
        .key_transform("audio", normalize)
        .shuffle()
        .batch(batch_size)
    )
    return data_iter


def eval_fn(model, inp, tgt):
    with torch.no_grad():
        return torch.sum(torch.argmax(model(inp), axis=1) == tgt) / tgt.shape[0]


def train_epoch(model, train_iter, optimizer, epoch, device):
    losses = []
    accs = []
    samples_per_sec = []
    model.train()
    for batch_counter, batch in enumerate(train_iter):
        x = torch.from_numpy(batch["audio"]).to(device)
        y = torch.from_numpy(batch["label"]).to(device)
        tic = time.perf_counter()
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = torch.sum(torch.argmax(output, axis=1) == y) / y.shape[0]
        loss = loss.item()
        acc = acc.item()
        toc = time.perf_counter()
        losses.append(loss)
        accs.append(acc)
        throughput = x.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        if batch_counter % 25 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {loss:.3f}",
                        f"Train acc {acc:.3f}",
                        f"Throughput: {throughput:.2f} samples/second",
                    )
                )
            )

    mean_tr_loss = torch.mean(torch.Tensor(losses))
    mean_tr_acc = torch.mean(torch.Tensor(accs))
    samples_per_sec = torch.mean(torch.Tensor(samples_per_sec))
    return mean_tr_loss, mean_tr_acc, samples_per_sec


def test_epoch(model, test_iter, device):
    model.eval()
    accs = []
    samples_per_sec = []
    for batch_counter, batch in enumerate(test_iter):
        x = torch.from_numpy(batch["audio"]).to(device)
        y = torch.from_numpy(batch["label"]).to(device)
        tic = time.perf_counter()
        acc = eval_fn(model, x, y)
        acc_value = acc.item()
        toc = time.perf_counter()
        throughput = x.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        accs.append(acc_value)
    mean_acc = torch.mean(torch.Tensor(accs))
    samples_per_sec = torch.mean(torch.Tensor(samples_per_sec))
    return mean_acc, samples_per_sec


def main(args):
    torch.manual_seed(args.seed)

    model = getattr(kwt_pytorch, args.arch)()
    model = model.to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    train_data = prepare_dataset(args.batch_size, "train")
    val_data = prepare_dataset(args.batch_size, "validation")

    best_params = None
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        tr_loss, tr_acc, throughput = train_epoch(
            model, train_data, optimizer, epoch, args.device
        )
        print(
            " | ".join(
                (
                    f"Epoch: {epoch}",
                    f"avg. Train loss {tr_loss.item():.3f}",
                    f"avg. Train acc {tr_acc.item():.3f}",
                    f"Throughput: {throughput.item():.2f} samples/sec",
                )
            )
        )

        val_acc, val_throughput = test_epoch(model, val_data, args.device)
        print(
            f"Epoch: {epoch} | Val acc {val_acc.item():.3f} | Throughput: {val_throughput.item():.2f} samples/sec"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_params = model.parameters()
    print(f"Testing best model from epoch {best_epoch}")
    model.load_state_dict(best_params)
    test_data = prepare_dataset(args.batch_size, "test")
    test_acc, _ = test_epoch(model, test_data, args.device)
    print(f"Test acc -> {test_acc.item():.3f}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    args.device = device
    main(args)
