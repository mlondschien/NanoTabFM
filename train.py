import time

import click
import torch
from torch.nn import functional as F

from model import NanoTabFM
from prior import make_batch


@click.command()
@click.option("--steps", default=3000)
@click.option("--batch-size", default=32)
@click.option("--learning-rate", default=3e-4)
@click.option("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
@click.option("--seed", default=0)
@click.option("--output", default="model.pt")
def main(steps, batch_size, learning_rate, device, seed, output):
    """Pretrain NanoTabFM on synthetic tables."""
    # Fixed validation tasks of several shapes, sampled on the CPU so they depend on neither
    # seed nor device.
    torch.manual_seed(0)
    validation_batches = [
        [tensor.to(device) for tensor in make_batch(64, num_context, 128, num_features)]
        for num_features, num_context in [(2, 50), (5, 200), (10, 400), (10, 800)]
    ]

    torch.manual_seed(seed)
    model = NanoTabFM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"NanoTabFM: {num_parameters:,} parameters, device={device}", flush=True)

    best_loss = float("inf")
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        num_features = int(torch.randint(1, 11, ()))
        # Few distinct shapes, because MPS builds new kernels for every new shape.
        num_context = 50 * int(torch.randint(1, 17, ()))
        context_features, context_labels, query_features, query_labels = make_batch(
            batch_size, num_context, 128, num_features, device=device
        )
        logits = model(context_features, context_labels, query_features)
        loss = F.cross_entropy(logits.flatten(0, 1), query_labels.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if device == "mps":
            # With varying shapes, cached MPS buffers otherwise outgrow physical memory.
            torch.mps.empty_cache()

        if step % 100 == 0 or step == steps:
            model.eval()
            losses, accuracies = [], []
            with torch.no_grad():
                for *inputs, query_labels in validation_batches:
                    logits = model(*inputs)
                    losses.append(F.cross_entropy(logits.flatten(0, 1), query_labels.flatten()))
                    accuracies.append((logits.argmax(dim=-1) == query_labels).float().mean())
            validation_loss = torch.stack(losses).mean().item()
            accuracy = torch.stack(accuracies).mean().item()
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(model.state_dict(), output)
            print(
                f"Step {step:5d} | train loss={loss.item():.4f} | "
                f"validation loss={validation_loss:.4f}, accuracy={accuracy:.1%} | "
                f"{time.perf_counter() - start_time:.1f}s",
                flush=True,
            )
    print(f"Best validation loss={best_loss:.4f}; saved to {output}")


if __name__ == "__main__":
    main()
