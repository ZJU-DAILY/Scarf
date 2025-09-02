import subprocess


def run_command(command, **kwargs):
    """Run a command while printing the live output"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    lines = []
    while True:  # Could be more pythonic with := in Python3.8+
        if process.poll() is not None:
            break
        line = process.stdout.readline()
        if not line:
            break
        print(line.decode(), end="")
        lines.append(line.decode())
    return "".join(lines)
