"""S3 read/write utilities for the sequence architect pipeline."""

import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init S3 client."""
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client


def read_file(bucket: str, key: str) -> str:
    """Read a text file from S3. Returns the decoded string."""
    s3 = _get_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def read_json(bucket: str, key: str) -> dict:
    """Read and parse a JSON file from S3."""
    return json.loads(read_file(bucket, key))


def read_file_optional(bucket: str, key: str) -> str | None:
    """Read a text file from S3, returning None if it does not exist."""
    try:
        return read_file(bucket, key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise


def write_json(bucket: str, key: str, data: dict) -> None:
    """Write a JSON object to S3."""
    s3 = _get_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )
    logger.debug("Wrote JSON to s3://%s/%s", bucket, key)


def write_text(bucket: str, key: str, text: str) -> None:
    """Write a text string to S3."""
    s3 = _get_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text,
        ContentType="text/plain",
    )
    logger.debug("Wrote text to s3://%s/%s", bucket, key)


def load_knowledge(bucket: str, key: str) -> str:
    """Load a knowledge file (markdown or text) from S3.

    Alias for read_file with clearer intent for prompt loading.
    """
    return read_file(bucket, key)
