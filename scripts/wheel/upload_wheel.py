# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Update the wheels page, prune old nightly builds if necessary (source from tlcpack)."""
import github3
import github3.session as session
import os
import logging
import argparse


def prune(args, path):
    gh = github3.GitHub(
        token=os.environ["GITHUB_TOKEN"],
        session=session.GitHubSession(default_connect_timeout=100, default_read_timeout=100),
    )
    repo = gh.repository(*args.repo.split("/"))
    release = repo.release_from_tag(args.tag)

    paths = []
    if os.path.isdir(path):
        for name in os.listdir(args.path):
            if name.endswith(".whl"):
                paths.append(os.path.join(path, name))
    else:
        paths = [path]

    for path in paths:
        name = os.path.basename(path)
        pkg_name = name[: name.find("-")]

        for asset in release.assets():
            if asset.name.startswith(pkg_name):
                if not args.dry_run:
                    asset.delete()
                print(f"Pruned old package: {asset.name}")


def upload(args, path):
    gh = github3.GitHub(
        token=os.environ["GITHUB_TOKEN"],
        session=session.GitHubSession(default_connect_timeout=100, default_read_timeout=100),
    )
    repo = gh.repository(*args.repo.split("/"))
    release = repo.release_from_tag(args.tag)

    paths = []
    if os.path.isdir(args.path):
        for name in os.listdir(args.path):
            if name.endswith(".whl"):
                paths.append(os.path.join(args.path, name))
    else:
        paths = [path]

    for path in paths:
        name = os.path.basename(path)
        content_bytes = open(path, "rb").read()

        for asset in release.assets():
            if asset.name == name:
                if not args.dry_run:
                    asset.delete()
                print(f"Removed duplicated package {asset.name}")

        print(f"Start to upload {path} to {args.repo}, this can take a while...")
        if not args.dry_run:
            release.upload_asset("application/octet-stream", name, content_bytes)
        print(f"Finished uploading {path}")


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Upload wheel as an asset of a tag.")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--repo", type=str, default="awslabs/raf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--path", type=str)

    if "GITHUB_TOKEN" not in os.environ:
        raise RuntimeError("GITHUB_TOKEN is unset")
    args = parser.parse_args()
    if args.prune:
        prune(args, args.path)
    upload(args, args.path)


if __name__ == "__main__":
    main()
