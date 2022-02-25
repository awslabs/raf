# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper tool to check license header."""
import os
import sys
import subprocess

usage = """
Usage: python3 scripts/lint/check_license_header.py <commit>

Run license header check that changed since <commit>
Examples:
 - Compare last one commit: python3 scripts/lint/check_license_header.py HEAD~1
 - Compare against upstream/main: python3 scripts/lint/check_license_header.py upstream/main
"""


def copyright_line(line):
    # Following two items are intentionally break apart
    # so that the copyright detector won't detect the file itself.
    if line.find("Copyright " + "(c)") != -1:
        return True
    # break pattern into two lines to avoid false-negative check
    spattern1 = "Copyright"
    if line.find(spattern1) != -1 and line.find("by") != -1:
        return True
    return False


def check_license(fname):
    if fname in [".gitignore", "LICENSE"]:
        return True

    # Skip 3rdparty change.
    if not os.path.isfile(fname):
        return True

    # Skip ignorable files.
    if fname.endswith(".png") or fname.endswith(".txt") or fname.find("LICENSE") != -1:
        return True

    has_license_header = False
    has_copyright = False
    try:
        for line in open(fname):
            if line.find("SPDX-License-Identifier: Apache-2.0") != -1:
                has_license_header = True
            elif line.find("Copyright Amazon.com, Inc.") != -1:
                has_copyright = True
            if has_license_header and has_copyright:
                return True
    except UnicodeDecodeError:
        pass
    return False


def main():
    if len(sys.argv) != 2:
        sys.stderr.write(usage)
        sys.stderr.flush()
        sys.exit(-1)

    arg = sys.argv[1]
    cmd = ["git", "diff", "--name-only", "--diff-filter=ACMRTUX", arg]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    assert proc.returncode == 0, f'{" ".join(cmd)} errored: {out}'
    res = out.decode("utf-8")

    error_list = []
    for fname in res.split():
        if not check_license(fname):
            error_list.append(fname)

    if error_list:
        report = "-----Check report-----\n"
        report += "\n".join(error_list) + "\n"
        report += "-----Found %d files that cannot pass the license header check-----\n" % len(
            error_list
        )
        report += "--- You can use the following steps to add the the license header:\n"
        report += "--- Create file_list.txt in your text editor\n"
        report += "--- Copy paste the above content in file-list into file_list.txt\n"
        report += "--- python3 scripts/lint/add_license_header.py file_list.txt\n"
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("check_license_header.py: all checks passed..")


if __name__ == "__main__":
    main()
