import os

import paramiko
from paramiko.channel import ChannelFile
from paramiko.client import SSHClient


def connect(client: SSHClient, hostname: str, **cfg) -> None:
    """
    Connect to a remote host using SSH.

    :param client: SSH client
    :param hostname: hostname
    :param cfg: configuration (username, port, etc.)
    :return:
    """
    client._policy = paramiko.WarningPolicy()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_config = paramiko.SSHConfig()
    user_config_file = os.path.expanduser("~/.ssh/config")
    if os.path.exists(user_config_file):
        with open(user_config_file) as f:
            ssh_config.parse(f)

    user_config = ssh_config.lookup(hostname)
    for k in ("hostname", "username", "port"):
        if k in user_config:
            cfg[k] = user_config[k]

    if "proxycommand" in user_config:
        cfg["sock"] = paramiko.ProxyCommand(user_config["proxycommand"])

    client.connect(**cfg)


def get_interactive_shell(
    client: SSHClient,
) -> tuple[ChannelFile, ChannelFile, ChannelFile]:
    channel = client.invoke_shell()
    stdin = channel.makefile("w")
    stdout = channel.makefile("r")
    stderr = channel.makefile_stderr("r")

    return stdin, stdout, stderr
