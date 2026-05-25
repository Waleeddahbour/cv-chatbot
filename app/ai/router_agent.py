from .agent_factory import create_registered_agent


def create_router_agent():
    return create_registered_agent("router")
