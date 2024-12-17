from cli.view_logs import view_logs

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    argumments = parse_arguments()

    if argumments.command == "logs":
        view_logs(
            model_name=argumments.model,
            tail=argumments.tail,
            lines=argumments.lines,
            follow=argumments.follow,
        )


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)


def parse_arguments():
    parser = ArgumentParser(description="The command line interface for MasInt")

    # Add a subparser for the 'logs' command
    subparsers = parser.add_subparsers(dest="command")

    logs_parser = subparsers.add_parser("logs", help="View logs")
    logs_parser.add_argument("--model", help="The model to view logs for")
    logs_parser.add_argument(
        "--tail",
        help="Whether to tail the logs",
        default=False,
        action="store_true",
    )
    logs_parser.add_argument(
        "--follow",
        help="Whether to follow the logs",
        default=False,
        action="store_true",
    )

    logs_parser.add_argument(
        "--lines", type=int, help="The number of lines to view", default=100
    )

    args = parser.parse_args()

    return args


main()
