"""Tests for the LLM factory and RoutingClient."""

import pytest

from personalab.config.loader import ProjectConfig
from personalab.llm.client import (
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)
from personalab.llm.factory import RoutingClient, create_client


# ---------------------------------------------------------------------------
# Minimal stub adapters (no vendor SDKs required)
# ---------------------------------------------------------------------------

class _StubText:
    def generate_text_json(self, **kw):
        return LLMTextResponse(parsed={"from": "text"})

    def generate_image(self, **kw):
        raise NotImplementedError

    def generate_video(self, **kw):
        raise NotImplementedError


class _StubImage:
    def generate_text_json(self, **kw):
        raise NotImplementedError

    def generate_image(self, **kw):
        return LLMImageResponse(images=[b"img"])

    def generate_video(self, **kw):
        raise NotImplementedError


class _StubVideo:
    def generate_text_json(self, **kw):
        raise NotImplementedError

    def generate_image(self, **kw):
        raise NotImplementedError

    def generate_video(self, **kw):
        return LLMVideoResponse(operation_name="vid-1")


# ---------------------------------------------------------------------------
# RoutingClient
# ---------------------------------------------------------------------------

class TestRoutingClient:
    def test_routes_text(self):
        rc = RoutingClient(text=_StubText(), image=_StubImage(), video=_StubVideo())
        resp = rc.generate_text_json(
            system_instruction="s", user_prompt="u",
            schema={}, use_search=False, model_name="m",
        )
        assert resp.parsed == {"from": "text"}

    def test_routes_image(self):
        rc = RoutingClient(text=_StubText(), image=_StubImage(), video=_StubVideo())
        resp = rc.generate_image(parts=[], aspect_ratio="1:1", model_name="m")
        assert resp.images == [b"img"]

    def test_routes_video(self):
        rc = RoutingClient(text=_StubText(), image=_StubImage(), video=_StubVideo())
        resp = rc.generate_video(
            prompt="p", resolution="1080p", aspect_ratio="9:16", model_name="m",
        )
        assert resp.operation_name == "vid-1"

    def test_unsupported_method_on_stub_raises(self):
        """Calling an unsupported modality directly on a stub raises."""
        with pytest.raises(NotImplementedError):
            _StubImage().generate_text_json(
                system_instruction="s", user_prompt="u",
                schema={}, use_search=False, model_name="m",
            )
        with pytest.raises(NotImplementedError):
            _StubText().generate_video(
                prompt="p", resolution="1080p", aspect_ratio="9:16", model_name="m",
            )


# ---------------------------------------------------------------------------
# create_client (factory function)
# ---------------------------------------------------------------------------

class TestCreateClient:
    def test_unknown_provider_raises(self):
        cfg = ProjectConfig(raw={
            "models": {
                "text": {"provider": "nonexistent", "model_name": "x"},
                "image": {"provider": "gemini", "model_name": "y"},
                "video": {"provider": "gemini", "model_name": "z"},
            }
        })
        with pytest.raises(ValueError, match="Unknown LLM provider 'nonexistent'"):
            create_client(cfg)

    def test_all_same_provider_returns_single_adapter(self, monkeypatch):
        """When all modalities use the same provider, a single adapter (not RoutingClient) is returned."""
        from personalab.llm import factory as fmod
        calls = []

        def _fake_gemini(**kw):
            calls.append(kw)
            return _StubText()

        monkeypatch.setitem(fmod._REGISTRY, "gemini", _fake_gemini)

        cfg = ProjectConfig(raw={
            "models": {
                "text": {"provider": "gemini", "model_name": "a"},
                "image": {"provider": "gemini", "model_name": "b"},
                "video": {"provider": "gemini", "model_name": "c"},
            }
        })
        client = create_client(cfg)
        assert not isinstance(client, RoutingClient)
        assert len(calls) == 1

    def test_mixed_providers_returns_routing_client(self, monkeypatch):
        from personalab.llm import factory as fmod

        monkeypatch.setitem(fmod._REGISTRY, "prov_text", lambda **kw: _StubText())
        monkeypatch.setitem(fmod._REGISTRY, "prov_image", lambda **kw: _StubImage())
        monkeypatch.setitem(fmod._REGISTRY, "prov_video", lambda **kw: _StubVideo())

        cfg = ProjectConfig(raw={
            "models": {
                "text": {"provider": "prov_text", "model_name": "t"},
                "image": {"provider": "prov_image", "model_name": "i"},
                "video": {"provider": "prov_video", "model_name": "v"},
            }
        })
        client = create_client(cfg)
        assert isinstance(client, RoutingClient)
        assert client.generate_text_json(
            system_instruction="", user_prompt="", schema={},
            use_search=False, model_name="",
        ).parsed == {"from": "text"}
        assert client.generate_image(parts=[], aspect_ratio="1:1", model_name="").images == [b"img"]
        assert client.generate_video(
            prompt="", resolution="", aspect_ratio="", model_name="",
        ).operation_name == "vid-1"

    def test_shared_provider_reuses_instance(self, monkeypatch):
        """Two modalities with the same provider key should share one adapter instance."""
        from personalab.llm import factory as fmod
        instances = []

        def _make(**kw):
            obj = _StubText()
            instances.append(obj)
            return obj

        monkeypatch.setitem(fmod._REGISTRY, "shared", _make)
        monkeypatch.setitem(fmod._REGISTRY, "other", lambda **kw: _StubVideo())

        cfg = ProjectConfig(raw={
            "models": {
                "text": {"provider": "shared", "model_name": "a"},
                "image": {"provider": "shared", "model_name": "b"},
                "video": {"provider": "other", "model_name": "c"},
            }
        })
        create_client(cfg)
        assert len(instances) == 1
