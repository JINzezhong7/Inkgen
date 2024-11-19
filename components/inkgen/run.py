import argparse
import subprocess
import shlex
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='directory to load training data')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--cell_size', type=int, default=512,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='rms',
                        help='optimizer to use (rms or adam)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use_scheduler', type=str, default='true',
                        help='whether or not to use LR scheduler')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='number of warmup steps')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='lr decay rate for adam optimizer per epoch')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of attention clusters on text input')
    parser.add_argument('--z_size', type=int, default=256,
                        help='style distribution size')
    parser.add_argument('--clip_value', type=float, default=100,
                        help='value to which to clip non-LSTM gradients')
    parser.add_argument('--lstm_clip_value', type=float, default=10,
                        help='value to which to clip LSTM gradients')
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                        help='checkpoint path from which to resume training')
    parser.add_argument('--style_equalization', type=str, default='false',
                        help='whether or not to train with style equalization')
    args = parser.parse_args()

    cmd = [
    "python", "src/train.py",
    "--data_dir", args.data_dir,
    "--model_dir", args.model_dir,
    "--cell_size", str(args.cell_size),
    "--batch_size", str(args.batch_size),
    "--num_epochs", str(args.num_epochs),
    "--optimizer", args.optimizer,
    "--learning_rate", str(args.learning_rate),
    "--warmup_steps", str(args.warmup_steps),
    "--decay_rate", str(args.decay_rate),
    "--num_clusters", str(args.num_clusters),
    "--K", str(args.K),
    "--z_size", str(args.z_size),
    "--clip_value", str(args.clip_value),
    "--lstm_clip_value", str(args.lstm_clip_value)
    ]

    if args.resume_from_ckpt.lower() != "none":
        cmd.extend(["--resume_from_ckpt", args.resume_from_ckpt])
    if args.use_scheduler.lower() == 'true':
        cmd.append("--use_scheduler")
    if args.style_equalization.lower() == 'true':
        cmd.append("--style_equalization")
    print(shlex.join(cmd))
    subprocess.run(shlex.join(cmd), shell=True)

