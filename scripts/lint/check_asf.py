# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Acknowledgement: The main logic originates from TVM

"""Helper tool to check ASF header."""
import os
import sys
import subprocess

usage = """
Usage: python3 scripts/lint/check_asf.py <commit>

Run ASF header check that changed since <commit>
Examples:
 - Compare last one commit: python3 scripts/lint/check_asf.py HEAD~1
 - Compare against upstream/main: python3 scripts/lint/check_asf.py upstream/main
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


def check_asf_copyright(fname):
    if fname in [
        ".gitignore",
        "LICENSE",
        "licenses/LICENSE.cutlass.txt",
        "licenses/LICENSE.googletest.txt",
        "licenses/LICENSE.tvm.txt",
    ]:
        return True
    if not os.path.isfile(fname):  # skip 3rdparty change
        return True
    if fname.endswith(".png"):  # skip binary files
        return True
    has_asf_header = False
    has_copyright = False
    try:
        for line in open(fname):
            if line.find("Licensed to the Apache Software Foundation") != -1:
                has_asf_header = True
            if copyright_line(line):
                has_copyright = True
            if has_asf_header and not has_copyright:
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

    asf_copyright_list = []
    for fname in res.split():
        if not check_asf_copyright(fname):
            asf_copyright_list.append(fname)

    if asf_copyright_list:
        report = "-----File ASF check report-----\n"
        report += "\n".join(asf_copyright_list) + "\n"
        report += "-----Found %d files that cannot pass the rough ASF header check-----\n" % len(
            asf_copyright_list
        )
        report += "--- Files should have ASF header and do not need Copyright lines.\n"
        report += "--- Contributors retain copyright to their contribution by default.\n"
        report += "--- If a file comes with a different license, consider put it under the 3rdparty folder instead.\n"
        report += "---\n"
        report += "--- You can use the following steps to remove the copyright lines\n"
        report += "--- Create file_list.txt in your text editor\n"
        report += "--- Copy paste the above content in file-list into file_list.txt\n"
        report += "--- python3 3rdparty/tvm/tests/lint/add_asf_header.py file_list.txt\n"
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("check_asf.py: all checks passed..")


if __name__ == "__main__":
    main()
