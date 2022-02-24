<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# CI on AWS Batch

We are going to deprecate Jenkins CI due to the security issue. Instead, we use Github Action as the CI agent with AWS Batch as the computation resource management. In this README, we document the behaviors and features of this mechanism.

## Flow Description

Here we introduce the CI flow when a PR has been filed.

### Linting

The first workflow `CI-Lint` will be triggered to lint the code changes. Note that since this is a normal Github Action workflow that checks out the PR for testing, we cannot make use of AWS credential in this workflow for security issue.

### Unit Test

After `CI-Lint` is done, it triggers the second workflow `CI-UnitTest`, which sets up AWS credential to submit jobs to AWS Batch. Since the actual testing happens on the Batch instance, this workflow does not have to checkout the PR. Instead, it only checks out the main branch to get the scripts in this folder. *It means if someone wants to modify the scripts in this folder, CI still uses the old one in main branch.*

Then, `CI-UnitTest` evaluates whether the rest unit tests can be skipped by checking whether the PR only changes documents or docker files.

If we have to run the rest tests, 3 more Github Action jobs will be created for CPU, GPU, and multi-GPU platforms. Each job will perform the following steps:

1. **Determine the job definition revision by platform name and docker image name.** It scans the existing revisions to find the one with matched docker image version. If the docker image version is used by the CI for the first time, a new revision of job definition will be created.
2. **Terminate previous running jobs.** If users submit a new commit to a PR, a new CI will be triggered and makes the existing running one useless. To save the computation resources, we terminate existing running jobs with the same job name, which is composed of the repository name and PR number.
3. **Submit a job and query logs.** Then it submits a new job to AWS Batch and monitors the job status. Specifically, it queries the job status every 10 seconds and prints to the console, so that we can directly see them from Github Action UI. AWS Batch job has the following status:
    - **SUBMITTED:** The job is just submitted.
    - **RUNNABLE:** The job configuration has no problem and can be executed. At this step, AWS Batch is configuring EC2 and ECS cluster for allocating this job, so it might be the case that job is RUNNABLE forever if 1) AWS Batch cannot launch the desired EC2 instances, 2) or the job definition includes too much memory (e.g., request exactly 32GB memory for 8 vCPUs).
    - **STARTING:** The instance is set up and the job is kicked off. If job is here, it is usually because AWS Batch is pulling the docker image, so it may take a while if the docker image is large.
    - **RUNNING**: The job is running. At this moment, we should be able to start seeing the log from AWS CloudWatch log stream.
    - **SUCCEEDED**: The job is done with exit code 0.
    - **FAILED**: The job is done with non-zero exit code. The log will also show the failed reason.
4. **Reuse CCache cache.**  Since compilation takes time, we backup the ccache cache to a S3 bucket and reuse it for the same PR (i.e., AWS Batch jobs with the same name).

## AWS Batch Configuration

AWS Batch has to be properly configured to make the above flow working as expected. In this document, we do dive into details about how to configure AWS Batch. Instead, we provide an example configuration for reference:

- Compute Environment
```
"computeEnvironments": [
    {
        "computeEnvironmentName": "ci-gpu",
        "computeEnvironmentArn": ***,
        "ecsClusterArn": ***,
        "tags": {
            "Name": "Batch - ci_gpu"
        },
        "type": "MANAGED",
        "state": "ENABLED",
        "status": "VALID",
        "statusReason": "ComputeEnvironment Healthy",
        "computeResources": {
            "type": "EC2",
            "allocationStrategy": "BEST_FIT",
            "minvCpus": 0,
            "maxvCpus": 256,
            "desiredvCpus": 48,
            "instanceTypes": [
                "g4dn"
            ],
            "subnets": [
                ***
            ],
            "securityGroupIds": [
                ***
            ],
            "instanceRole": ***,
            "tags": {
                "Name": "Batch - ci-gpu"
            },
            "ec2Configuration": [
                {
                    "imageType": "ECS_AL2_NVIDIA"
                }
            ]
        },
        "serviceRole": ***
    },
    {
        "computeEnvironmentName": "ci-cpu",
        "computeEnvironmentArn": ***,
        "ecsClusterArn": ***,
        "tags": {
            "Name": "Batch - ci-cpu"
        },
        "type": "MANAGED",
        "state": "ENABLED",
        "status": "VALID",
        "statusReason": "ComputeEnvironment Healthy",
        "computeResources": {
            "type": "EC2",
            "allocationStrategy": "BEST_FIT",
            "minvCpus": 0,
            "maxvCpus": 256,
            "desiredvCpus": 0,
            "instanceTypes": [
                "optimal"
            ],
            "subnets": [
                ***
            ],
            "securityGroupIds": [
                ***
            ],
            "instanceRole": ***,
            "tags": {
                "Name": "Batch - ci-cpu"
            },
            "ec2Configuration": [
                {
                    "imageType": "ECS_AL2"
                }
            ]
        },
        "serviceRole": ***
    }
]
```

- Job Queue

```
"jobQueues": [
    {
        "jobQueueName": "ci-cpu-queue",
        "jobQueueArn": ***,
        "state": "ENABLED",
        "status": "VALID",
        "statusReason": "JobQueue Healthy",
        "priority": 1,
        "computeEnvironmentOrder": [
            {
                "order": 1,
                "computeEnvironment": "arn:aws:batch:***:compute-environment/ci-cpu"
            }
        ],
        "tags": {}
    },
    {
        "jobQueueName": "ci-gpu-queue",
        "jobQueueArn": ***,
        "state": "ENABLED",
        "status": "VALID",
        "statusReason": "JobQueue Healthy",
        "priority": 1,
        "computeEnvironmentOrder": [
            {
                "order": 1,
                "computeEnvironment": "arn:aws:batch:***:compute-environment/ci-gpu"
            }
        ],
        "tags": {}
    }
]
```

- Job Definition

```
"jobDefinitions": [
    {
        "jobDefinitionName": "ci-job-meta-gpu",
        "jobDefinitionArn": "arn:aws:batch:***:job-definition/ci-job-meta-gpu:5",
        "revision": 5,
        "status": "ACTIVE",
        "type": "container",
        "parameters": {},
        "containerProperties": {
            "image": "metaprojdev/raf:ci_gpu-v0.20",
            "command": [],
            "jobRoleArn": ***,
            "executionRoleArn": ***,
            "volumes": [],
            "environment": [],
            "mountPoints": [],
            "ulimits": [],
            "resourceRequirements": [
                {
                    "value": "16",
                    "type": "VCPU"
                },
                {
                    "value": "61440",
                    "type": "MEMORY"
                },
                {
                    "value": "1",
                    "type": "GPU"
                }
            ],
            "secrets": [
                {
                    "name": "GITHUB_TOKEN",
                    "valueFrom": "arn:aws:secretsmanager:***:secret:ci/meta/github_token-***:GITHUB_TOKEN::"
                }
            ]
        },
        "tags": {},
        "propagateTags": false,
        "platformCapabilities": [
            "EC2"
        ]
    },
    {
        "jobDefinitionName": "ci-job-meta-cpu",
        "jobDefinitionArn": "arn:aws:batch:***:job-definition/ci-job-meta-cpu:7",
        "revision": 7,
        "status": "ACTIVE",
        "type": "container",
        "parameters": {},
        "containerProperties": {
            "image": "metaprojdev/raf:ci_cpu-v0.18",
            "command": [],
            "jobRoleArn": ***,
            "executionRoleArn": ***,
            "volumes": [],
            "environment": [],
            "mountPoints": [],
            "ulimits": [],
            "resourceRequirements": [
                {
                    "value": "8",
                    "type": "VCPU"
                },
                {
                    "value": "30720",
                    "type": "MEMORY"
                }
            ],
            "secrets": [
                {
                    "name": "GITHUB_TOKEN",
                    "valueFrom": "arn:aws:secretsmanager:***:secret:ci/meta/github_token-***:GITHUB_TOKEN::"
                }
            ]
        },
        "tags": {},
        "propagateTags": false,
        "platformCapabilities": [
            "EC2"
        ]
    },
    {
        "jobDefinitionName": "ci-job-meta-multi-gpu",
        "jobDefinitionArn": "arn:aws:batch:***:job-definition/ci-job-meta-multi-gpu:1",
        "revision": 1,
        "status": "ACTIVE",
        "type": "container",
        "parameters": {},
        "containerProperties": {
            "image": "metaprojdev/raf:ci_gpu-v0.20",
            "command": [],
            "jobRoleArn": ***,
            "executionRoleArn": ***,
            "volumes": [],
            "environment": [],
            "mountPoints": [],
            "ulimits": [],
            "resourceRequirements": [
                {
                    "value": "8",
                    "type": "VCPU"
                },
                {
                    "value": "30720",
                    "type": "MEMORY"
                },
                {
                    "value": "2",
                    "type": "GPU"
                }
            ],
            "secrets": [
                {
                    "name": "GITHUB_TOKEN",
                    "valueFrom": "arn:aws:secretsmanager:***:secret:ci/meta/github_token-***:GITHUB_TOKEN::"
                }
            ]
        },
        "tags": {},
        "propagateTags": false,
        "platformCapabilities": [
            "EC2"
        ]
    }
]
```

## AWS Batch Scripts for CI

The CI scripts for AWS Batch integration.

### cli.sh

A set of bash shell utility functions for CIs, including cmake configuration, compilation, and unit test executions.

### submit-job.py

A Python script to submit CI jobs to AWS Batch.

### job-def-cfg.json

The JSON configurataion file to specify the mapping from unit test platform to the corresponding AWS Batch job definition. Note that job definition revision will be realized by `submit-job.py` and does not have to be specified in this configuration file.

