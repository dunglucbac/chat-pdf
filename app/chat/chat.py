import random
from typing import Any, Callable

from langchain.chat_models import ChatOpenAI
from app.chat.models import ChatArgs, ComponentType
from app.chat.vector_stores import retriever_map
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.web.api import set_conversation_components, get_conversation_components
from app.chat.score import random_component_by_score
from app.chat.tracing.langfuse_client import langfuse_instance


def select_component(
    component_type: ComponentType,
    component_map: dict[str, Callable[[ChatArgs], Any]],
    chat_args: ChatArgs,
) -> tuple[str, Any]:
    """
    The function first checks whether the conversation already has a saved
    component name for the given component_type (for example: llm, retriever,
    or memory). If a previous component exists, it reuses that component to
    keep behavior consistent within the conversation.

    If no component has been saved yet, it chooses one using weighted random
    selection based on historical scores, then builds that component.

    Args:
        component_type (ComponentType): The component category to resolve.
            One of ComponentType.LLM, ComponentType.RETRIEVER, or ComponentType.MEMORY.
        component_map (dict[str, Callable[[ChatArgs], Any]]): Mapping of
            component name to builder function.
        chat_args (ChatArgs): Conversation context used by builder functions.

    Returns:
        tuple[str, Any]: A tuple containing:
            - selected component name
            - instantiated component object

    Raises:
        KeyError: If component_type is missing from stored conversation
            components, or if the selected component name does not exist
            in component_map.
        ValueError: If score-based selection receives an invalid component_type
            (propagated from random_component_by_score).
    """
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]

    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    else:
        random_name = random_component_by_score(component_type, component_map)
        builder = component_map[random_name]
        return random_name, builder(chat_args)


def build_chat(chat_args: ChatArgs) -> StreamingConversationalRetrievalChain:
    retriever_name, retriever = select_component(ComponentType.RETRIEVER, retriever_map, chat_args)
    llm_name, llm = select_component(ComponentType.LLM, llm_map, chat_args)
    memory_name, memory = select_component(ComponentType.MEMORY, memory_map, chat_args)
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory=memory_name,
    )

    condense_question_llm = ChatOpenAI(streaming=False)

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_question_llm,
        memory=memory,
        retriever=retriever,
        metadata=chat_args.metadata,
    )
