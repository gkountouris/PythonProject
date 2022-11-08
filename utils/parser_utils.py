import argparse

from utils import utils

def add_data_arguments(parser):
    parser.add_argument("--train_path", type=str, help="Train path.", required=False)
    parser.add_argument("--dev_path", type=str, help="Dev path.", required=False)
    parser.add_argument("--test_path", type=str, help="Test path.", required=False)
    parser.add_argument("--keep_only", type=str, default='factoid_snippet', help="Keep Only.", required=False)

def add_encoder_arguments(parser):
    parser.add_argument("--model_name", type=str, default="michiyasunaga/BioLinkBERT-large", help="Prefix.",
                        required=False)
    parser.add_argument("--transformer_size", type=int, default=1024, help="Prefix.", required=False)

def add_optimization_arguments(parser):
    parser.add_argument("--batch_size", type=int, default=16, nargs="?", help="Batch size.", required=False)
    parser.add_argument("--warmup_steps", type=int, default=0, nargs="?", help="Warmup steps.", required=False)
    parser.add_argument("--total_epochs", type=int, default=50, nargs="?", help="Total epochs.", required=False)
    parser.add_argument("--patience", type=int, default=5, nargs="?", help="patience.", required=False)
    parser.add_argument("--hidden_dim", type=int, default=100, nargs="?", help="Hidden dimensions.", required=False)
    parser.add_argument("--lr", type=float, default=5e-5, nargs="?", help="Learning rate.", required=False)
    parser.add_argument("--monitor", type=str, default='auc', help="loss OR auc.", required=False)

def add_additional_arguments(parser):
    parser.add_argument("--seed", type=int, default=1, nargs="?", help="Random seed.", required=False)
    parser.add_argument("--prefix", type=str, help="Prefix.", required=False)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser


