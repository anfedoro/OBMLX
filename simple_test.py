import mlx.core as mx
import mlx.optimizers as optim

import matplotlib.pyplot as plt
from mlx.utils import tree_flatten
import time

import subprocess

import mlx.nn as nn
import math
from obf.obaf import QubOBAF
from tensorboardX import SummaryWriter

dtype = mx.float32  # Set default dtype to bfloat16 for MLX


def test_function_0(x):
    y_true_1 = mx.cos(x**2) - mx.sin(x * 3) / 3 + mx.cos(x * 6) / 4 - mx.sin(x * 5) / 5
    y_true_2 = mx.sin(x * -0.5) - mx.sin(x * 3 * -0.5) / 3
    y_true = mx.cos(y_true_1 + y_true_2)
    return y_true


def test_function_1(x):
    y_true_1 = mx.sin(x) + mx.cos(x**2) - mx.cos(x * 3) / 3 + mx.cos(x * 4) / 4 - mx.sin(x * 5) / 5
    y_true_2 = mx.sin(x * -0.5) + mx.cos(x**2 * -0.5) / 2 - mx.sin(x * 3 * -0.5) / 3
    y_true = mx.cos(y_true_1 + y_true_2)
    return y_true


def test_function_2(x):
    y_true_1 = mx.sin(x) - mx.sin(x * 3) / 3 + mx.cos(x * 4) / 4 - mx.sin(x * 5) / 5
    y_true_2 = mx.cos(x**2 * -0.5) / 2 - mx.sin(x * 3 * -0.5) / 3
    y_true = mx.sin(y_true_1 + y_true_2**2)
    return y_true


def test_function_3(x):
    y_true_1 = mx.sin(x) + mx.cos(x**2) - mx.sin(x * 3) / 3 + mx.cos(x * 4) / 4 - mx.cos(x * 3) / 5
    y_true_2 = mx.sin(x * -0.5) + mx.cos(x**2 * -0.5) / 3
    y_true = y_true_1 * y_true_2
    return y_true


# def artificial_2d_function(x):
#     return torch.sin(x[0]) * torch.cos(x[1]) + 0.5 * torch.sin(2 * x[0] + x[1])


# Using MLX built-in MSE loss function instead of custom implementation
def test_function(x):
    y_0 = test_function_0(x[..., 0:1, :])
    y_1 = test_function_1(x[..., 1:2, :])
    y_2 = test_function_2(x[..., 2:3, :])
    y_3 = test_function_3(x[..., 3:4, :])
    y = mx.sin(y_0 / 2 + y_1) - mx.cos(y_2 / 3 + y_3) + mx.sin(y_0 * 2) - mx.cos(y_1 * 3)
    return y


def main():
    mx.set_default_device(mx.gpu)

    mx.random.seed(42)
    N = 500  # Number of points in the input space
    train_samples = 100
    eval_samples = 25
    # TensorBoard writer and fixed random eval set for evolution monitoring
    num_functions = 4  # Number of functions to test
    x = mx.zeros((train_samples, num_functions, N)).astype(dtype)  # [100, 4, N]
    for i in range(train_samples):
        x[i, :, :] = mx.sort(mx.random.uniform(shape=(N,)) * 2 * math.pi - math.pi).astype(dtype)

    y = test_function(x)  # [100, 4, N]

    # prepare fixed random eval x
    x_eval = mx.zeros((eval_samples, num_functions, N)).astype(dtype)  # [25, 1, N]
    for i in range(eval_samples):
        x_eval[i, :, :] = mx.sort(mx.random.uniform(shape=(N,)) * 2 * math.pi - math.pi).astype(dtype)
    # x_eval = x_eval.expand(funcs, N)
    y_eval = test_function(x_eval)

    experiment_name = "obaf_mx_14"
    subprocess.run(f"rm -rf runs/{experiment_name}", shell=True)
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
    model = nn.Sequential(
        QubOBAF(inp_dim=4, out_dim=8, size=3, stride=2, dropout=0.01),
        QubOBAF(inp_dim=8, out_dim=8, size=3, stride=2, dropout=0.01),
        QubOBAF(inp_dim=8, out_dim=1, size=3, stride=2, dropout=0.01),
    )

    print(model)
    model.set_dtype(dtype)
    # Count parameters using MLX method
    num_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    print(f"Number of parameters in the model: {num_params}")

    initial_lr = 5e-4
    final_lr = 1e-5
    total_epochs = 20000

    # Use MLX built-in cosine decay scheduler
    lr_schedule = optim.cosine_decay(initial_lr, total_epochs // 2, final_lr)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # Define loss function once
    def loss_fn(model):
        return nn.losses.mse_loss(model(x), y)

    # Create the loss_and_grad function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    best_eval_loss = float("inf")
    target_loss = 1e-5
    target_epoch = None
    tick = time.time()
    for epoch in range(total_epochs):
        # Training step
        loss, grads = loss_and_grad(model)
        optim.clip_grad_norm(grads, max_norm=1.0)  # Clip gradients to prevent exploding gradients
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Evaluation
        y_pred_eval = model(x_eval)
        eval_loss = nn.losses.mse_loss(y_pred_eval, y_eval)

        # Convert to float for logging
        loss_val = float(loss)
        eval_loss_val = float(eval_loss)

        # TensorBoard logging
        writer.add_scalars("Loss", {"Train": loss_val, "Eval": eval_loss_val}, global_step=epoch)
        writer.add_scalar("Learning Rate", optimizer.learning_rate.item(), global_step=epoch)

        if epoch % 20 == 0:
            plt.rcParams.update({"font.size": 6})
            fig, ax = plt.subplots()
            fig.set_size_inches(16, 8)

            # Convert MLX arrays to numpy for plotting
            x1 = mx.mean(x, axis=0).mean(axis=0, keepdims=True).squeeze().astype(mx.float32)
            y1 = mx.mean(y, axis=0).squeeze().astype(mx.float32)
            y_pred = model(x).astype(mx.float32)
            y_pred = mx.mean(y_pred, axis=0).squeeze().astype(mx.float32)
            y_pred_eval = mx.mean(y_pred_eval, axis=0).squeeze().astype(mx.float32)

            ax.plot(x1, y1, label="True", linewidth=1)
            ax.plot(x1, y_pred, linestyle="dotted", label="Pred", linewidth=1)
            ax.plot(x1, y_pred_eval, linestyle="dotted", label="Pred Eval", linewidth=1)
            ax.set_title(f"Epoch {epoch}")
            ax.legend()
            writer.add_figure("Evolution/RandomEval", fig, global_step=epoch)
            plt.close(fig)

        if eval_loss_val < best_eval_loss:
            best_eval_loss = eval_loss_val
            best_loss = loss_val
            if not target_epoch:
                if eval_loss_val < target_loss:
                    target_epoch = epoch
                    break

        ratio = (epoch + 1) / (time.time() - tick)
        print(
            f"Epoch {epoch} \tLoss: {loss_val:.6f} \tEval Loss: {eval_loss_val:.6f} \tLR: {optimizer.learning_rate.item():.6f}\tRatio: {ratio:.2f} it/s",
            end="\r",
            flush=True,
        )

        if epoch % (total_epochs // 10) == 0:
            print(f"Epoch {epoch} \tLoss: {loss_val:.6f} \tEval Loss: {eval_loss_val:.6f} \tLR: {optimizer.learning_rate.item():.6f}")

    print(f"Best Loss: {best_loss}, Best eval loss: {best_eval_loss} Epoch with target loss: {target_epoch}")
    writer.flush()
    writer.close()
    # Final evaluation and plotting
    y_pred = model(x)
    y_pred_eval = model(x_eval)

    plt.figure(figsize=(16, 8))
    xx = mx.mean(x_eval, axis=0).mean(axis=0, keepdims=True).squeeze()
    x1 = mx.mean(x, axis=0).mean(axis=0, keepdims=True).squeeze()
    y1 = mx.mean(y, axis=0).squeeze()
    y_pred = mx.mean(y_pred, axis=0).squeeze()
    y_pred_eval = mx.mean(y_pred_eval, axis=0).squeeze()

    print(f"xx shape {xx.shape}, x1 shape {x1.shape}, y1 shape {y1.shape}, y_pred shape {y_pred.shape}, y_pred_eval shape {y_pred_eval.shape}")

    plt.plot(mx.array(x1), mx.array(y_pred), label="Pred", linestyle="--")
    plt.plot(mx.array(x1), mx.array(y1), label="True")
    plt.plot(mx.array(x1), mx.array(y_pred_eval), label="Pred Eval", linestyle=":")

    plt.legend()
    plt.grid()
    plt.title("Predictions")
    plt.show()
    writer.add_figure("Final Predictions", plt.gcf())


if __name__ == "__main__":
    main()
