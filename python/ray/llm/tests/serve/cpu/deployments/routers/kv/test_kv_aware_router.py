import sys

import pytest

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.ingress.builder import (
    LLMServingArgs,
    build_openai_app,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KV_ROUTER_ACTOR_NAME,
    KVRouterActor,
)
from ray.serve.llm.request_router import KVAwareRouter

# NOTE: Tests that deploy the KVRouterActor (which needs ai-dynamo, intentionally
# not installed in this CI) live in the llm_kv_router release test under
# release/llm_tests/kv_router_test/. This covers only the dynamo-free check that
# a KVAwareRouter attaches the deployment actor config.


def get_kv_actor_configs(deployment):
    return [
        cfg
        for cfg in (deployment._deployment_config.deployment_actors or [])
        if (cfg["name"] if isinstance(cfg, dict) else cfg.name) == KV_ROUTER_ACTOR_NAME
    ]


def build_test_llm_config() -> LLMConfig:
    return LLMConfig(
        model_loading_config={
            "model_id": "qwen3-0.6b",
            "model_source": "Qwen/Qwen3-0.6B",
        },
        accelerator_type=None,
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
            "request_router_config": {"request_router_class": KVAwareRouter},
        },
    )


@pytest.fixture(autouse=True)
def enable_direct_streaming(monkeypatch):
    monkeypatch.setattr(
        "ray.llm._internal.serve.core.ingress.builder."
        "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
        True,
    )


def test_build_openai_app_attaches_kv_actor():
    """A KVAwareRouter on the LLMConfig attaches the KVRouterActor."""
    app = build_openai_app(LLMServingArgs(llm_configs=[build_test_llm_config()]))

    configs = get_kv_actor_configs(app._bound_deployment)
    assert len(configs) == 1
    actor_cfg = configs[0]
    assert (
        actor_cfg.get_actor_class().__ray_actor_class__
        is KVRouterActor.__ray_actor_class__
    )
    assert actor_cfg.actor_options["num_cpus"] == 0
    assert actor_cfg.init_kwargs == {"block_size": 16}


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
