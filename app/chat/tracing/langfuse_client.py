import os
from langfuse.client import Langfuse
from langfuse.model import CreateTrace

langfuse_instance = Langfuse(
    os.environ["LANGFUSE_PUBLIC_KEY"],
    os.environ["LANGFUSE_SECRET_KEY"],
    host="https://prod-langfuse.fly.dev",
)
