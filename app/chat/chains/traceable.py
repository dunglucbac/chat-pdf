from langfuse.model import CreateTrace
from app.chat.tracing.langfuse_client import langfuse_instance


class TraceableChain:
    def __call__(self, *args, **kwds):
        trace = langfuse_instance.trace(
            CreateTrace(
                id=self.metadata["conversation_id"],
                metadata=self.metadata,
            )
        )
        callbacks = kwds.get("callbacks", [])
        callbacks.append(trace.getNewHandler())
        kwds["callbacks"] = callbacks
        return super().__call__(*args, **kwds)
