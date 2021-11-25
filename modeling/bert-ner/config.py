import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="ner", help="ner/pos/chunk")
    parser.add_argument("--dataset_name", type=str, default="conll2003")

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--label_all_tokens", default=True)

    parser.add_argument("--metric_strategy", type=str, default="seqeval")
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()
    return args
